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




struct MinkowskiCenter_f7efefbe
{
    wp::vec_t<3, wp::float32> B;
    wp::vec_t<3, wp::float32> BtoA;


    MinkowskiCenter_f7efefbe() = default;
    CUDA_CALLABLE MinkowskiCenter_f7efefbe(wp::vec_t<3, wp::float32> const& B,
    wp::vec_t<3, wp::float32> const& BtoA = {})
        : B{B}
        , BtoA{BtoA}

    {
    }

    CUDA_CALLABLE MinkowskiCenter_f7efefbe& operator += (const MinkowskiCenter_f7efefbe& rhs)
    {    B += rhs.B;
    BtoA += rhs.BtoA;

        return *this;}

};

static CUDA_CALLABLE void adj_MinkowskiCenter_f7efefbe(wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> & adj_B,
    wp::vec_t<3, wp::float32> & adj_BtoA,
    MinkowskiCenter_f7efefbe & adj_ret)
{
    adj_B += adj_ret.B;
    adj_BtoA += adj_ret.BtoA;
}

// Required when compiling adjoints.
CUDA_CALLABLE MinkowskiCenter_f7efefbe add(const MinkowskiCenter_f7efefbe& a, const MinkowskiCenter_f7efefbe& b)
{
    return MinkowskiCenter_f7efefbe();
}

CUDA_CALLABLE void adj_atomic_add(MinkowskiCenter_f7efefbe* p, MinkowskiCenter_f7efefbe t)
{
    wp::adj_atomic_add(&p->B, t.B);
    wp::adj_atomic_add(&p->BtoA, t.BtoA);
}




struct Vert_99873389
{
    wp::vec_t<3, wp::float32> B;
    wp::vec_t<3, wp::float32> BtoA;


    Vert_99873389() = default;
    CUDA_CALLABLE Vert_99873389(wp::vec_t<3, wp::float32> const& B,
    wp::vec_t<3, wp::float32> const& BtoA = {})
        : B{B}
        , BtoA{BtoA}

    {
    }

    CUDA_CALLABLE Vert_99873389& operator += (const Vert_99873389& rhs)
    {    B += rhs.B;
    BtoA += rhs.BtoA;

        return *this;}

};

static CUDA_CALLABLE void adj_Vert_99873389(wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> & adj_B,
    wp::vec_t<3, wp::float32> & adj_BtoA,
    Vert_99873389 & adj_ret)
{
    adj_B += adj_ret.B;
    adj_BtoA += adj_ret.BtoA;
}

// Required when compiling adjoints.
CUDA_CALLABLE Vert_99873389 add(const Vert_99873389& a, const Vert_99873389& b)
{
    return Vert_99873389();
}

CUDA_CALLABLE void adj_atomic_add(Vert_99873389* p, Vert_99873389 t)
{
    wp::adj_atomic_add(&p->B, t.B);
    wp::adj_atomic_add(&p->BtoA, t.BtoA);
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




struct IncrementalPlaneTracker_8af6ef33
{
    wp::vec_t<3, wp::float32> reference_point;
    wp::vec_t<3, wp::float32> previous_point;
    wp::vec_t<3, wp::float32> normal;
    wp::float32 largest_area_sq;


    IncrementalPlaneTracker_8af6ef33() = default;
    CUDA_CALLABLE IncrementalPlaneTracker_8af6ef33(wp::vec_t<3, wp::float32> const& reference_point,
    wp::vec_t<3, wp::float32> const& previous_point = {},
    wp::vec_t<3, wp::float32> const& normal = {},
    wp::float32 const& largest_area_sq = {})
        : reference_point{reference_point}
        , previous_point{previous_point}
        , normal{normal}
        , largest_area_sq{largest_area_sq}

    {
    }

    CUDA_CALLABLE IncrementalPlaneTracker_8af6ef33& operator += (const IncrementalPlaneTracker_8af6ef33& rhs)
    {    reference_point += rhs.reference_point;
    previous_point += rhs.previous_point;
    normal += rhs.normal;
    largest_area_sq += rhs.largest_area_sq;

        return *this;}

};

static CUDA_CALLABLE void adj_IncrementalPlaneTracker_8af6ef33(wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::float32 const&,
    wp::vec_t<3, wp::float32> & adj_reference_point,
    wp::vec_t<3, wp::float32> & adj_previous_point,
    wp::vec_t<3, wp::float32> & adj_normal,
    wp::float32 & adj_largest_area_sq,
    IncrementalPlaneTracker_8af6ef33 & adj_ret)
{
    adj_reference_point += adj_ret.reference_point;
    adj_previous_point += adj_ret.previous_point;
    adj_normal += adj_ret.normal;
    adj_largest_area_sq += adj_ret.largest_area_sq;
}

// Required when compiling adjoints.
CUDA_CALLABLE IncrementalPlaneTracker_8af6ef33 add(const IncrementalPlaneTracker_8af6ef33& a, const IncrementalPlaneTracker_8af6ef33& b)
{
    return IncrementalPlaneTracker_8af6ef33();
}

CUDA_CALLABLE void adj_atomic_add(IncrementalPlaneTracker_8af6ef33* p, IncrementalPlaneTracker_8af6ef33 t)
{
    wp::adj_atomic_add(&p->reference_point, t.reference_point);
    wp::adj_atomic_add(&p->previous_point, t.previous_point);
    wp::adj_atomic_add(&p->normal, t.normal);
    wp::adj_atomic_add(&p->largest_area_sq, t.largest_area_sq);
}




struct BodyProjector_d067bb7a
{
    wp::float32 plane_d;
    wp::vec_t<3, wp::float32> normal;


    BodyProjector_d067bb7a() = default;
    CUDA_CALLABLE BodyProjector_d067bb7a(wp::float32 const& plane_d,
    wp::vec_t<3, wp::float32> const& normal = {})
        : plane_d{plane_d}
        , normal{normal}

    {
    }

    CUDA_CALLABLE BodyProjector_d067bb7a& operator += (const BodyProjector_d067bb7a& rhs)
    {    plane_d += rhs.plane_d;
    normal += rhs.normal;

        return *this;}

};

static CUDA_CALLABLE void adj_BodyProjector_d067bb7a(wp::float32 const&,
    wp::vec_t<3, wp::float32> const&,
    wp::float32 & adj_plane_d,
    wp::vec_t<3, wp::float32> & adj_normal,
    BodyProjector_d067bb7a & adj_ret)
{
    adj_plane_d += adj_ret.plane_d;
    adj_normal += adj_ret.normal;
}

// Required when compiling adjoints.
CUDA_CALLABLE BodyProjector_d067bb7a add(const BodyProjector_d067bb7a& a, const BodyProjector_d067bb7a& b)
{
    return BodyProjector_d067bb7a();
}

CUDA_CALLABLE void adj_atomic_add(BodyProjector_d067bb7a* p, BodyProjector_d067bb7a t)
{
    wp::adj_atomic_add(&p->plane_d, t.plane_d);
    wp::adj_atomic_add(&p->normal, t.normal);
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:768
static CUDA_CALLABLE void extract_shape_data_0(
    wp::int32 var_shape_idx,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::int32> var_shape_types,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::uint64> var_shape_source,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::quat_t<wp::float32> & ret_1,
    GenericShapeData_ba895271 & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::float32 & ret_4)
{
    //---------
    // primal vars
    wp::transform_t<wp::float32>* var_0;
    wp::transform_t<wp::float32> var_1;
    wp::transform_t<wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::quat_t<wp::float32> var_4;
    wp::vec_t<4, wp::float32>* var_5;
    wp::vec_t<4, wp::float32> var_6;
    wp::vec_t<4, wp::float32> var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 3;
    wp::float32 var_16;
    GenericShapeData_ba895271 var_17;
    wp::int32* var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    const wp::float32 var_21 = 0.0;
    const wp::float32 var_22 = 0.0;
    const wp::float32 var_23 = 0.0;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 0.0;
    wp::vec_t<3, wp::float32> var_28;
    wp::int32* var_29;
    const wp::int32 var_30 = 10;
    bool var_31;
    wp::int32 var_32;
    wp::uint64* var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::uint64 var_35;
    //---------
    // forward
    // def extract_shape_data(                                                                <L 769>
    // X_ws = shape_transform[shape_idx]                                                      <L 790>
    var_0 = wp::address(var_shape_transform, var_shape_idx);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // position = wp.transform_get_translation(X_ws)                                          <L 792>
    var_3 = wp::transform_get_translation(var_1);
    // orientation = wp.transform_get_rotation(X_ws)                                          <L 793>
    var_4 = wp::transform_get_rotation(var_1);
    // data = shape_data[shape_idx]                                                           <L 797>
    var_5 = wp::address(var_shape_data, var_shape_idx);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // scale = wp.vec3(data[0], data[1], data[2])                                             <L 798>
    var_9 = wp::extract(var_6, var_8);
    var_11 = wp::extract(var_6, var_10);
    var_13 = wp::extract(var_6, var_12);
    var_14 = wp::vec_t<3, wp::float32>(var_9, var_11, var_13);
    // margin_offset = data[3]                                                                <L 799>
    var_16 = wp::extract(var_6, var_15);
    // result = GenericShapeData()                                                            <L 802>
    var_17 = GenericShapeData_ba895271();
    // result.shape_type = shape_types[shape_idx]                                             <L 803>
    var_18 = wp::address(var_shape_types, var_shape_idx);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    var_17.shape_type = var_19;
    // result.scale = scale                                                                   <L 804>
    var_17.scale = var_14;
    // result.auxiliary = wp.vec3(0.0, 0.0, 0.0)                                              <L 805>
    var_24 = wp::vec_t<3, wp::float32>(var_21, var_22, var_23);
    var_17.auxiliary = var_24;
    // result.center = wp.vec3(0.0, 0.0, 0.0)                                                 <L 806>
    var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
    var_17.center = var_28;
    // result.shape_index = shape_idx                                                         <L 807>
    var_17.shape_index = var_shape_idx;
    // if shape_types[shape_idx] == GeoType.CONVEX_MESH:                                      <L 810>
    var_29 = wp::address(var_shape_types, var_shape_idx);
    var_32 = wp::load(var_29);
    var_31 = (var_32 == var_30);
    if (var_31) {
        // result.auxiliary = pack_mesh_ptr(shape_source[shape_idx])                          <L 811>
        var_33 = wp::address(var_shape_source, var_shape_idx);
        var_35 = wp::load(var_33);
        var_34 = pack_mesh_ptr_0(var_35);
        var_17.auxiliary = var_34;
    }
    // return position, orientation, result, scale, margin_offset                             <L 813>
    ret_0 = var_3;
    ret_1 = var_4;
    ret_2 = var_17;
    ret_3 = var_14;
    ret_4 = var_16;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:516
static CUDA_CALLABLE void compute_bounding_sphere_from_aabb_0(
    wp::vec_t<3, wp::float32> var_aabb_lower,
    wp::vec_t<3, wp::float32> var_aabb_upper,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::float32 var_3 = 0.5;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    //---------
    // forward
    // def compute_bounding_sphere_from_aabb(aabb_lower: wp.vec3, aabb_upper: wp.vec3) -> tuple[wp.vec3, float]:       <L 517>
    // center = 0.5 * (aabb_lower + aabb_upper)                                               <L 524>
    var_1 = wp::add(var_aabb_lower, var_aabb_upper);
    var_2 = wp::mul(var_0, var_1);
    // half_extents = 0.5 * (aabb_upper - aabb_lower)                                         <L 525>
    var_4 = wp::sub(var_aabb_upper, var_aabb_lower);
    var_5 = wp::mul(var_3, var_4);
    // radius = wp.length(half_extents)                                                       <L 526>
    var_6 = wp::length(var_5);
    // return center, radius                                                                  <L 527>
    ret_0 = var_2;
    ret_1 = var_6;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:594
static CUDA_CALLABLE bool check_infinite_plane_bsphere_overlap_0(
    GenericShapeData_ba895271 var_shape_data_a,
    GenericShapeData_ba895271 var_shape_data_b,
    wp::vec_t<3, wp::float32> var_pos_a,
    wp::vec_t<3, wp::float32> var_pos_b,
    wp::quat_t<wp::float32> var_quat_a,
    wp::quat_t<wp::float32> var_quat_b,
    wp::vec_t<3, wp::float32> var_bsphere_center_a,
    wp::vec_t<3, wp::float32> var_bsphere_center_b,
    wp::float32 var_bsphere_radius_a,
    wp::float32 var_bsphere_radius_b)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::int32* var_3;
    wp::int32 var_4;
    wp::int32 var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    bool var_12;
    const wp::int32 var_13 = 1;
    bool var_14;
    bool var_15;
    const wp::int32 var_16 = 0;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.0;
    bool var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    bool var_24;
    const wp::int32 var_25 = 1;
    bool var_26;
    bool var_27;
    const wp::int32 var_28 = 0;
    wp::float32 var_29;
    const wp::float32 var_30 = 0.0;
    bool var_31;
    const wp::int32 var_32 = 1;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0;
    bool var_35;
    bool var_36;
    bool var_37;
    const bool var_38 = true;
    wp::vec_t<3, wp::float32> var_39;
    wp::quat_t<wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::quat_t<wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::float32 var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::quat_t<wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::float32 var_50;
    const wp::float32 var_51 = 0.0;
    const wp::float32 var_52 = 0.0;
    const wp::float32 var_53 = 1.0;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    bool var_58;
    //---------
    // forward
    // def check_infinite_plane_bsphere_overlap(                                              <L 595>
    // type_a = shape_data_a.shape_type                                                       <L 614>
    var_0 = &((var_shape_data_a).shape_type);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // type_b = shape_data_b.shape_type                                                       <L 615>
    var_3 = &((var_shape_data_b).shape_type);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // scale_a = shape_data_a.scale                                                           <L 616>
    var_6 = &((var_shape_data_a).scale);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // scale_b = shape_data_b.scale                                                           <L 617>
    var_9 = &((var_shape_data_b).scale);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // is_infinite_plane_a = (type_a == GeoType.PLANE) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)       <L 620>
    var_14 = (var_1 == var_13);
    var_12 = var_14;
    if (var_12) {
        var_17 = wp::extract(var_7, var_16);
        var_19 = (var_17 == var_18);
        var_15 = var_19;
        if (var_15) {
            var_21 = wp::extract(var_7, var_20);
            var_23 = (var_21 == var_22);
            var_15 = var_15 && var_23;
        }
        var_12 = var_12 && var_15;
    }
    // is_infinite_plane_b = (type_b == GeoType.PLANE) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)       <L 621>
    var_26 = (var_4 == var_25);
    var_24 = var_26;
    if (var_24) {
        var_29 = wp::extract(var_10, var_28);
        var_31 = (var_29 == var_30);
        var_27 = var_31;
        if (var_27) {
            var_33 = wp::extract(var_10, var_32);
            var_35 = (var_33 == var_34);
            var_27 = var_27 && var_35;
        }
        var_24 = var_24 && var_27;
    }
    // if not (is_infinite_plane_a or is_infinite_plane_b):                                   <L 624>
    var_36 = var_12;
    if (!var_36) {
        var_36 = var_36 || var_24;
    }
    var_37 = wp::unot(var_36);
    if (var_37) {
        // return True                                                                        <L 625>
        return var_38;
    }
    // if is_infinite_plane_a:                                                                <L 628>
    if (var_12) {
        // plane_pos = pos_a                                                                  <L 629>
        var_39 = wp::copy(var_pos_a);
        // plane_quat = quat_a                                                                <L 630>
        var_40 = wp::copy(var_quat_a);
        // other_center = bsphere_center_b                                                    <L 631>
        var_41 = wp::copy(var_bsphere_center_b);
        // other_radius = bsphere_radius_b                                                    <L 632>
        var_42 = wp::copy(var_bsphere_radius_b);
    }
    if (!var_12) {
        // plane_pos = pos_b                                                                  <L 634>
        var_43 = wp::copy(var_pos_b);
        // plane_quat = quat_b                                                                <L 635>
        var_44 = wp::copy(var_quat_b);
        // other_center = bsphere_center_a                                                    <L 636>
        var_45 = wp::copy(var_bsphere_center_a);
        // other_radius = bsphere_radius_a                                                    <L 637>
        var_46 = wp::copy(var_bsphere_radius_a);
    }
    var_47 = wp::where(var_12, var_39, var_43);
    var_48 = wp::where(var_12, var_40, var_44);
    var_49 = wp::where(var_12, var_41, var_45);
    var_50 = wp::where(var_12, var_42, var_46);
    // plane_normal = wp.quat_rotate(plane_quat, wp.vec3(0.0, 0.0, 1.0))                      <L 640>
    var_54 = wp::vec_t<3, wp::float32>(var_51, var_52, var_53);
    var_55 = wp::quat_rotate(var_48, var_54);
    // center_dist = wp.dot(other_center - plane_pos, plane_normal)                           <L 643>
    var_56 = wp::sub(var_49, var_47);
    var_57 = wp::dot(var_56, var_55);
    // return center_dist <= other_radius                                                     <L 647>
    var_58 = (var_57 <= var_50);
    return var_58;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:530
static CUDA_CALLABLE void convert_infinite_plane_to_cube_0(
    GenericShapeData_ba895271 var_shape_data,
    wp::quat_t<wp::float32> var_plane_rotation,
    wp::vec_t<3, wp::float32> var_plane_position,
    wp::vec_t<3, wp::float32> var_other_position,
    wp::float32 var_other_radius,
    GenericShapeData_ba895271 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    GenericShapeData_ba895271 var_0;
    const wp::int32 var_1 = 7;
    const wp::float32 var_2 = 10.0;
    wp::float32 var_3;
    const wp::float32 var_4 = 10.0;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    const wp::float32 var_9 = 0.0;
    wp::vec_t<3, wp::float32> var_10;
    const wp::float32 var_11 = 0.0;
    const wp::float32 var_12 = 0.0;
    const wp::float32 var_13 = 0.0;
    wp::vec_t<3, wp::float32> var_14;
    const wp::float32 var_15 = 0.0;
    const wp::float32 var_16 = 0.0;
    const wp::float32 var_17 = 1.0;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::float32 var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    //---------
    // forward
    // def convert_infinite_plane_to_cube(                                                    <L 531>
    // result = GenericShapeData()                                                            <L 558>
    var_0 = GenericShapeData_ba895271();
    // result.shape_type = GeoType.BOX                                                        <L 559>
    var_0.shape_type = var_1;
    // lateral_size = other_radius * 10.0                                                     <L 564>
    var_3 = wp::mul(var_other_radius, var_2);
    // depth = other_radius * 10.0                                                            <L 568>
    var_5 = wp::mul(var_other_radius, var_4);
    // result.scale = wp.vec3(lateral_size, lateral_size, depth)                              <L 573>
    var_6 = wp::vec_t<3, wp::float32>(var_3, var_3, var_5);
    var_0.scale = var_6;
    // result.auxiliary = wp.vec3(0.0, 0.0, 0.0)                                              <L 574>
    var_10 = wp::vec_t<3, wp::float32>(var_7, var_8, var_9);
    var_0.auxiliary = var_10;
    // result.center = wp.vec3(0.0, 0.0, 0.0)                                                 <L 575>
    var_14 = wp::vec_t<3, wp::float32>(var_11, var_12, var_13);
    var_0.center = var_14;
    // plane_normal = wp.quat_rotate(plane_rotation, wp.vec3(0.0, 0.0, 1.0))                  <L 579>
    var_18 = wp::vec_t<3, wp::float32>(var_15, var_16, var_17);
    var_19 = wp::quat_rotate(var_plane_rotation, var_18);
    // to_other = other_position - plane_position                                             <L 580>
    var_20 = wp::sub(var_other_position, var_plane_position);
    // distance_along_normal = wp.dot(to_other, plane_normal)                                 <L 581>
    var_21 = wp::dot(var_20, var_19);
    // plane_surface_point = other_position - plane_normal * distance_along_normal            <L 584>
    var_22 = wp::mul(var_19, var_21);
    var_23 = wp::sub(var_other_position, var_22);
    // adjusted_position = plane_surface_point - plane_normal * depth                         <L 589>
    var_24 = wp::mul(var_19, var_5);
    var_25 = wp::sub(var_23, var_24);
    // return result, adjusted_position                                                       <L 591>
    ret_0 = var_0;
    ret_1 = var_25;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:816
static CUDA_CALLABLE wp::vec_t<3, wp::float32> closest_point_on_triangle_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_tri_a,
    wp::vec_t<3, wp::float32> var_tri_b,
    wp::vec_t<3, wp::float32> var_tri_c)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-20;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    bool var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    bool var_10;
    bool var_11;
    bool var_12;
    bool var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.0;
    const wp::float32 var_18 = 1.0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    bool var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 1.0;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0;
    const wp::float32 var_35 = 1.0;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    bool var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    const wp::float32 var_47 = 0.0;
    bool var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    bool var_52;
    const wp::float32 var_53 = 0.0;
    bool var_54;
    bool var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    bool var_59;
    const wp::float32 var_60 = 0.0;
    bool var_61;
    bool var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    bool var_66;
    const wp::float32 var_67 = 0.0;
    bool var_68;
    const wp::float32 var_69 = 0.0;
    bool var_70;
    const wp::float32 var_71 = 0.0;
    bool var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    bool var_80;
    const wp::float32 var_81 = 0.0;
    bool var_82;
    const wp::float32 var_83 = 0.0;
    bool var_84;
    const wp::float32 var_85 = 0.0;
    bool var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    bool var_94;
    const wp::float32 var_95 = 0.0;
    bool var_96;
    wp::float32 var_97;
    const wp::float32 var_98 = 0.0;
    bool var_99;
    wp::float32 var_100;
    const wp::float32 var_101 = 0.0;
    bool var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::float32 var_111;
    const wp::float32 var_112 = 1.0;
    wp::float32 var_113;
    wp::float32 var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::vec_t<3, wp::float32> var_119;
    wp::vec_t<3, wp::float32> var_120;
    wp::vec_t<3, wp::float32> var_121;
    //---------
    // forward
    // def closest_point_on_triangle(                                                         <L 817>
    // ab = tri_b - tri_a                                                                     <L 838>
    var_0 = wp::sub(var_tri_b, var_tri_a);
    // ac = tri_c - tri_a                                                                     <L 839>
    var_1 = wp::sub(var_tri_c, var_tri_a);
    // ab_sq = wp.dot(ab, ab)                                                                 <L 844>
    var_2 = wp::dot(var_0, var_0);
    // ac_sq = wp.dot(ac, ac)                                                                 <L 845>
    var_3 = wp::dot(var_1, var_1);
    // EPS2 = 1.0e-20                                                                         <L 846>
    // triangle_normal = wp.cross(ab, ac)                                                     <L 847>
    var_5 = wp::cross(var_0, var_1);
    // if wp.dot(triangle_normal, triangle_normal) < EPS2:                                    <L 848>
    var_6 = wp::dot(var_5, var_5);
    var_7 = (var_6 < var_4);
    if (var_7) {
        // bc = tri_c - tri_b                                                                 <L 849>
        var_8 = wp::sub(var_tri_c, var_tri_b);
        // bc_sq = wp.dot(bc, bc)                                                             <L 850>
        var_9 = wp::dot(var_8, var_8);
        // if ab_sq >= ac_sq and ab_sq >= bc_sq:                                              <L 851>
        var_11 = (var_2 >= var_3);
        var_10 = var_11;
        if (var_10) {
            var_12 = (var_2 >= var_9);
            var_10 = var_10 && var_12;
        }
        if (var_10) {
            // if ab_sq < EPS2:                                                               <L 852>
            var_13 = (var_2 < var_4);
            if (var_13) {
                // return tri_a                                                               <L 853>
                return var_tri_a;
            }
            // t = wp.clamp(wp.dot(p - tri_a, ab) / ab_sq, 0.0, 1.0)                          <L 854>
            var_14 = wp::sub(var_p, var_tri_a);
            var_15 = wp::dot(var_14, var_0);
            var_16 = wp::div(var_15, var_2);
            var_19 = wp::clamp(var_16, var_17, var_18);
            // return tri_a + t * ab                                                          <L 855>
            var_20 = wp::mul(var_19, var_0);
            var_21 = wp::add(var_tri_a, var_20);
            return var_21;
        }
        if (!var_10) {
            // elif ac_sq >= bc_sq:                                                           <L 856>
            var_22 = (var_3 >= var_9);
            if (var_22) {
                // t = wp.clamp(wp.dot(p - tri_a, ac) / ac_sq, 0.0, 1.0)                      <L 857>
                var_23 = wp::sub(var_p, var_tri_a);
                var_24 = wp::dot(var_23, var_1);
                var_25 = wp::div(var_24, var_3);
                var_28 = wp::clamp(var_25, var_26, var_27);
                // return tri_a + t * ac                                                      <L 858>
                var_29 = wp::mul(var_28, var_1);
                var_30 = wp::add(var_tri_a, var_29);
                return var_30;
            }
            if (!var_22) {
                // t = wp.clamp(wp.dot(p - tri_b, bc) / bc_sq, 0.0, 1.0)                      <L 860>
                var_31 = wp::sub(var_p, var_tri_b);
                var_32 = wp::dot(var_31, var_8);
                var_33 = wp::div(var_32, var_9);
                var_36 = wp::clamp(var_33, var_34, var_35);
                // return tri_b + t * bc                                                      <L 861>
                var_37 = wp::mul(var_36, var_8);
                var_38 = wp::add(var_tri_b, var_37);
                return var_38;
            }
            var_39 = wp::where(var_22, var_28, var_36);
        }
        var_40 = wp::where(var_10, var_19, var_39);
    }
    // ap = p - tri_a                                                                         <L 863>
    var_41 = wp::sub(var_p, var_tri_a);
    // d1 = wp.dot(ab, ap)                                                                    <L 865>
    var_42 = wp::dot(var_0, var_41);
    // d2 = wp.dot(ac, ap)                                                                    <L 866>
    var_43 = wp::dot(var_1, var_41);
    // if d1 <= 0.0 and d2 <= 0.0:                                                            <L 867>
    var_46 = (var_42 <= var_45);
    var_44 = var_46;
    if (var_44) {
        var_48 = (var_43 <= var_47);
        var_44 = var_44 && var_48;
    }
    if (var_44) {
        // return tri_a                                                                       <L 868>
        return var_tri_a;
    }
    // bp = p - tri_b                                                                         <L 870>
    var_49 = wp::sub(var_p, var_tri_b);
    // d3 = wp.dot(ab, bp)                                                                    <L 871>
    var_50 = wp::dot(var_0, var_49);
    // d4 = wp.dot(ac, bp)                                                                    <L 872>
    var_51 = wp::dot(var_1, var_49);
    // if d3 >= 0.0 and d4 <= d3:                                                             <L 873>
    var_54 = (var_50 >= var_53);
    var_52 = var_54;
    if (var_52) {
        var_55 = (var_51 <= var_50);
        var_52 = var_52 && var_55;
    }
    if (var_52) {
        // return tri_b                                                                       <L 874>
        return var_tri_b;
    }
    // cp = p - tri_c                                                                         <L 876>
    var_56 = wp::sub(var_p, var_tri_c);
    // d5 = wp.dot(ab, cp)                                                                    <L 877>
    var_57 = wp::dot(var_0, var_56);
    // d6 = wp.dot(ac, cp)                                                                    <L 878>
    var_58 = wp::dot(var_1, var_56);
    // if d6 >= 0.0 and d5 <= d6:                                                             <L 879>
    var_61 = (var_58 >= var_60);
    var_59 = var_61;
    if (var_59) {
        var_62 = (var_57 <= var_58);
        var_59 = var_59 && var_62;
    }
    if (var_59) {
        // return tri_c                                                                       <L 880>
        return var_tri_c;
    }
    // vc = d1 * d4 - d3 * d2                                                                 <L 882>
    var_63 = wp::mul(var_42, var_51);
    var_64 = wp::mul(var_50, var_43);
    var_65 = wp::sub(var_63, var_64);
    // if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:                                              <L 883>
    var_68 = (var_65 <= var_67);
    var_66 = var_68;
    if (var_66) {
        var_70 = (var_42 >= var_69);
        var_66 = var_66 && var_70;
    }
    if (var_66) {
        var_72 = (var_50 <= var_71);
        var_66 = var_66 && var_72;
    }
    if (var_66) {
        // v = d1 / (d1 - d3)                                                                 <L 884>
        var_73 = wp::sub(var_42, var_50);
        var_74 = wp::div(var_42, var_73);
        // return tri_a + v * ab                                                              <L 885>
        var_75 = wp::mul(var_74, var_0);
        var_76 = wp::add(var_tri_a, var_75);
        return var_76;
    }
    // vb = d5 * d2 - d1 * d6                                                                 <L 887>
    var_77 = wp::mul(var_57, var_43);
    var_78 = wp::mul(var_42, var_58);
    var_79 = wp::sub(var_77, var_78);
    // if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:                                              <L 888>
    var_82 = (var_79 <= var_81);
    var_80 = var_82;
    if (var_80) {
        var_84 = (var_43 >= var_83);
        var_80 = var_80 && var_84;
    }
    if (var_80) {
        var_86 = (var_58 <= var_85);
        var_80 = var_80 && var_86;
    }
    if (var_80) {
        // w = d2 / (d2 - d6)                                                                 <L 889>
        var_87 = wp::sub(var_43, var_58);
        var_88 = wp::div(var_43, var_87);
        // return tri_a + w * ac                                                              <L 890>
        var_89 = wp::mul(var_88, var_1);
        var_90 = wp::add(var_tri_a, var_89);
        return var_90;
    }
    // va = d3 * d6 - d5 * d4                                                                 <L 892>
    var_91 = wp::mul(var_50, var_58);
    var_92 = wp::mul(var_57, var_51);
    var_93 = wp::sub(var_91, var_92);
    // if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:                                <L 893>
    var_96 = (var_93 <= var_95);
    var_94 = var_96;
    if (var_94) {
        var_97 = wp::sub(var_51, var_50);
        var_99 = (var_97 >= var_98);
        var_94 = var_94 && var_99;
    }
    if (var_94) {
        var_100 = wp::sub(var_57, var_58);
        var_102 = (var_100 >= var_101);
        var_94 = var_94 && var_102;
    }
    if (var_94) {
        // w = (d4 - d3) / ((d4 - d3) + (d5 - d6))                                            <L 894>
        var_103 = wp::sub(var_51, var_50);
        var_104 = wp::sub(var_51, var_50);
        var_105 = wp::sub(var_57, var_58);
        var_106 = wp::add(var_104, var_105);
        var_107 = wp::div(var_103, var_106);
        // return tri_b + w * (tri_c - tri_b)                                                 <L 895>
        var_108 = wp::sub(var_tri_c, var_tri_b);
        var_109 = wp::mul(var_107, var_108);
        var_110 = wp::add(var_tri_b, var_109);
        return var_110;
    }
    var_111 = wp::where(var_94, var_107, var_88);
    // denom = 1.0 / (va + vb + vc)                                                           <L 897>
    var_113 = wp::add(var_93, var_79);
    var_114 = wp::add(var_113, var_65);
    var_115 = wp::div(var_112, var_114);
    // v = vb * denom                                                                         <L 898>
    var_116 = wp::mul(var_79, var_115);
    // w = vc * denom                                                                         <L 899>
    var_117 = wp::mul(var_65, var_115);
    // return tri_a + v * ab + w * ac                                                         <L 900>
    var_118 = wp::mul(var_116, var_0);
    var_119 = wp::add(var_tri_a, var_118);
    var_120 = wp::mul(var_117, var_1);
    var_121 = wp::add(var_119, var_120);
    return var_121;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _adjust_minkowski_center_0(
    GenericShapeData_ba895271 var_geom_a,
    wp::vec_t<3, wp::float32> var_center_b_world,
    wp::vec_t<3, wp::float32> var_center_b_to_a)
{
    //---------
    // primal vars
    bool var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 1000;
    const wp::int32 var_3 = 1000;
    wp::int32 var_4;
    bool var_5;
    wp::int32 var_6;
    wp::int32* var_7;
    const wp::int32 var_8 = 1001;
    const wp::int32 var_9 = 1001;
    wp::int32 var_10;
    bool var_11;
    wp::int32 var_12;
    const wp::float32 var_13 = 0.0;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32>* var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    const wp::float32 var_26 = 1e-20;
    bool var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::float32 var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    bool var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    bool var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::float32 var_45;
    const wp::float32 var_46 = 0.0;
    bool var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 0.0;
    bool var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::float32 var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    const wp::float32 var_62 = 3.0;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::float32 var_68;
    const wp::float32 var_69 = 1e-12;
    bool var_70;
    const wp::float32 var_71 = 0.01;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::int32* var_79;
    const wp::int32 var_80 = 1001;
    const wp::int32 var_81 = 1001;
    wp::int32 var_82;
    bool var_83;
    wp::int32 var_84;
    const wp::float32 var_85 = 0.0;
    const wp::float32 var_86 = 0.0;
    const wp::float32 var_87 = 0.5;
    const wp::float32 var_88 = 1.0;
    wp::float32 var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::vec_t<3, wp::float32> var_92;
    //---------
    // forward
    // def _adjust_minkowski_center(geom_a: Any, center_b_world: wp.vec3, center_b_to_a: wp.vec3) -> wp.vec3:       <L 1>
    // if geom_a.shape_type != int(GeoTypeEx.TRIANGLE) and geom_a.shape_type != int(GeoTypeEx.TRIANGLE_PRISM):       <L 3>
    var_1 = &((var_geom_a).shape_type);
    var_4 = wp::int(var_3);
    var_6 = wp::load(var_1);
    var_5 = (var_6 != var_4);
    var_0 = var_5;
    if (var_0) {
        var_7 = &((var_geom_a).shape_type);
        var_10 = wp::int(var_9);
        var_12 = wp::load(var_7);
        var_11 = (var_12 != var_10);
        var_0 = var_0 && var_11;
    }
    if (var_0) {
        // return center_b_to_a                                                               <L 4>
        return var_center_b_to_a;
    }
    // tri_a = wp.vec3(0.0)                                                                   <L 6>
    var_14 = wp::vec_t<3, wp::float32>(var_13);
    // tri_b = geom_a.scale                                                                   <L 7>
    var_15 = &((var_geom_a).scale);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // tri_c = geom_a.auxiliary                                                               <L 8>
    var_18 = &((var_geom_a).auxiliary);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    // face_normal = wp.cross(tri_b - tri_a, tri_c - tri_a)                                   <L 9>
    var_21 = wp::sub(var_16, var_14);
    var_22 = wp::sub(var_19, var_14);
    var_23 = wp::cross(var_21, var_22);
    // face_normal_length_sq = wp.length_sq(face_normal)                                      <L 10>
    var_24 = wp::length_sq(var_23);
    // projection = closest_point_on_triangle(center_b_world, tri_a, tri_b, tri_c)            <L 11>
    var_25 = closest_point_on_triangle_0(var_center_b_world, var_14, var_16, var_19);
    // if face_normal_length_sq < 1.0e-20:                                                    <L 12>
    var_27 = (var_24 < var_26);
    if (var_27) {
        // return projection - center_b_world                                                 <L 13>
        var_28 = wp::sub(var_25, var_center_b_world);
        return var_28;
    }
    // face_normal_unit = face_normal / wp.sqrt(face_normal_length_sq)                        <L 15>
    var_29 = wp::sqrt(var_24);
    var_30 = wp::div(var_23, var_29);
    // signed_plane_distance = wp.dot(center_b_world - tri_a, face_normal_unit)               <L 16>
    var_31 = wp::sub(var_center_b_world, var_14);
    var_32 = wp::dot(var_31, var_30);
    // plane_projection = center_b_world - signed_plane_distance * face_normal_unit           <L 17>
    var_33 = wp::mul(var_32, var_30);
    var_34 = wp::sub(var_center_b_world, var_33);
    // inside_face = (                                                                        <L 18>
    // wp.dot(wp.cross(tri_b - tri_a, plane_projection - tri_a), face_normal) >= 0.0          <L 19>
    var_36 = wp::sub(var_16, var_14);
    var_37 = wp::sub(var_34, var_14);
    var_38 = wp::cross(var_36, var_37);
    var_39 = wp::dot(var_38, var_23);
    var_41 = (var_39 >= var_40);
    var_35 = var_41;
    if (var_35) {
        // and wp.dot(wp.cross(tri_c - tri_b, plane_projection - tri_b), face_normal) >= 0.0       <L 20>
        var_42 = wp::sub(var_19, var_16);
        var_43 = wp::sub(var_34, var_16);
        var_44 = wp::cross(var_42, var_43);
        var_45 = wp::dot(var_44, var_23);
        var_47 = (var_45 >= var_46);
        var_35 = var_35 && var_47;
    }
    if (var_35) {
        // and wp.dot(wp.cross(tri_a - tri_c, plane_projection - tri_c), face_normal) >= 0.0       <L 21>
        var_48 = wp::sub(var_14, var_19);
        var_49 = wp::sub(var_34, var_19);
        var_50 = wp::cross(var_48, var_49);
        var_51 = wp::dot(var_50, var_23);
        var_53 = (var_51 >= var_52);
        var_35 = var_35 && var_53;
    }
    // if inside_face:                                                                        <L 23>
    if (var_35) {
        // projection = plane_projection                                                      <L 24>
        var_54 = wp::copy(var_34);
        // center_b_to_a = -signed_plane_distance * face_normal_unit                          <L 25>
        var_55 = wp::neg(var_32);
        var_56 = wp::mul(var_55, var_30);
    }
    if (!var_35) {
        // center_b_to_a = projection - center_b_world                                        <L 27>
        var_57 = wp::sub(var_25, var_center_b_world);
    }
    var_58 = wp::where(var_35, var_56, var_57);
    var_59 = wp::where(var_35, var_54, var_25);
    // to_centroid = (tri_a + tri_b + tri_c) / 3.0 - projection                               <L 29>
    var_60 = wp::add(var_14, var_16);
    var_61 = wp::add(var_60, var_19);
    var_63 = wp::div(var_61, var_62);
    var_64 = wp::sub(var_63, var_59);
    // to_centroid -= wp.dot(to_centroid, face_normal_unit) * face_normal_unit                <L 30>
    var_65 = wp::dot(var_64, var_30);
    var_66 = wp::mul(var_65, var_30);
    var_67 = wp::sub(var_64, var_66);
    // distance_to_centroid = wp.length(to_centroid)                                          <L 31>
    var_68 = wp::length(var_67);
    // if distance_to_centroid > 1.0e-12:                                                     <L 32>
    var_70 = (var_68 > var_69);
    if (var_70) {
        // nudge_distance = 0.01 * wp.min(distance_to_centroid, wp.abs(signed_plane_distance))       <L 33>
        var_72 = wp::abs(var_32);
        var_73 = wp::min(var_68, var_72);
        var_74 = wp::mul(var_71, var_73);
        // center_b_to_a += to_centroid * (nudge_distance / distance_to_centroid)             <L 34>
        var_75 = wp::div(var_74, var_68);
        var_76 = wp::mul(var_67, var_75);
        var_77 = wp::add(var_58, var_76);
    }
    var_78 = wp::where(var_70, var_77, var_58);
    // if geom_a.shape_type == int(GeoTypeEx.TRIANGLE_PRISM):                                 <L 36>
    var_79 = &((var_geom_a).shape_type);
    var_82 = wp::int(var_81);
    var_84 = wp::load(var_79);
    var_83 = (var_84 == var_82);
    if (var_83) {
        // center_b_to_a -= wp.vec3(0.0, 0.0, 0.5 * TRIANGLE_PRISM_EXTRUSION)                 <L 45>
        var_89 = wp::mul(var_87, var_88);
        var_90 = wp::vec_t<3, wp::float32>(var_85, var_86, var_89);
        var_91 = wp::sub(var_78, var_90);
    }
    var_92 = wp::where(var_83, var_91, var_78);
    // return center_b_to_a                                                                   <L 46>
    return var_92;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE MinkowskiCenter_f7efefbe create_shape_center_function__locals__shape_center_16(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    MinkowskiCenter_f7efefbe var_0;
    const bool var_1 = true;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32>* var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    //---------
    // forward
    // def shape_center(                                                                      <L 1>
    // center = MinkowskiCenter()                                                             <L 21>
    var_0 = MinkowskiCenter_f7efefbe();
    // if wp.static(use_precomputed_center):                                                  <L 22>
    // center_a = geom_a.center                                                               <L 23>
    var_2 = &((var_geom_a).center);
    var_4 = wp::load(var_2);
    var_3 = wp::copy(var_4);
    // center_b_local = geom_b.center                                                         <L 24>
    var_5 = &((var_geom_b).center);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // center.B = position_b + wp.quat_rotate(orientation_b, center_b_local)                  <L 29>
    var_8 = wp::quat_rotate(var_orientation_b, var_6);
    var_9 = wp::add(var_position_b, var_8);
    var_0.B = var_9;
    // center.BtoA = _adjust_minkowski_center(geom_a, center.B, center_a - center.B)          <L 30>
    var_10 = &((var_0).B);
    var_11 = &((var_0).B);
    var_13 = wp::load(var_11);
    var_12 = wp::sub(var_3, var_13);
    var_15 = wp::load(var_10);
    var_14 = _adjust_minkowski_center_0(var_geom_a, var_15, var_12);
    var_0.BtoA = var_14;
    // return center                                                                          <L 31>
    return var_0;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _minkowski_center_fallback_0(
    GenericShapeData_ba895271 var_geom_a,
    wp::vec_t<3, wp::float32> var_center_b_world)
{
    //---------
    // primal vars
    bool var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 1000;
    const wp::int32 var_3 = 1000;
    wp::int32 var_4;
    bool var_5;
    wp::int32 var_6;
    wp::int32* var_7;
    const wp::int32 var_8 = 1001;
    const wp::int32 var_9 = 1001;
    wp::int32 var_10;
    bool var_11;
    wp::int32 var_12;
    const wp::float32 var_13 = 0.0;
    wp::vec_t<3, wp::float32> var_14;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32>* var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::float32 var_26;
    const wp::float32 var_27 = 1e-20;
    bool var_28;
    const wp::float32 var_29 = 0.0;
    wp::vec_t<3, wp::float32> var_30;
    wp::float32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    const wp::float32 var_36 = 3.0;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::float32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::float32 var_45;
    const wp::float32 var_46 = 0.0;
    bool var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    const wp::float32 var_50 = 1e-20;
    bool var_51;
    const wp::float32 var_52 = 0.01;
    wp::vec_t<3, wp::float32> var_53;
    wp::float32 var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::vec_t<3, wp::float32> var_58;
    const wp::float32 var_59 = 1e-05;
    wp::vec_t<3, wp::float32> var_60;
    //---------
    // forward
    // def _minkowski_center_fallback(geom_a: Any, center_b_world: wp.vec3) -> wp.vec3:       <L 1>
    // if geom_a.shape_type != int(GeoTypeEx.TRIANGLE) and geom_a.shape_type != int(GeoTypeEx.TRIANGLE_PRISM):       <L 3>
    var_1 = &((var_geom_a).shape_type);
    var_4 = wp::int(var_3);
    var_6 = wp::load(var_1);
    var_5 = (var_6 != var_4);
    var_0 = var_5;
    if (var_0) {
        var_7 = &((var_geom_a).shape_type);
        var_10 = wp::int(var_9);
        var_12 = wp::load(var_7);
        var_11 = (var_12 != var_10);
        var_0 = var_0 && var_11;
    }
    if (var_0) {
        // return wp.vec3(0.0)                                                                <L 4>
        var_14 = wp::vec_t<3, wp::float32>(var_13);
        return var_14;
    }
    // tri_a = wp.vec3(0.0)                                                                   <L 6>
    var_16 = wp::vec_t<3, wp::float32>(var_15);
    // tri_b = geom_a.scale                                                                   <L 7>
    var_17 = &((var_geom_a).scale);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // tri_c = geom_a.auxiliary                                                               <L 8>
    var_20 = &((var_geom_a).auxiliary);
    var_22 = wp::load(var_20);
    var_21 = wp::copy(var_22);
    // face_normal = wp.cross(tri_b - tri_a, tri_c - tri_a)                                   <L 9>
    var_23 = wp::sub(var_18, var_16);
    var_24 = wp::sub(var_21, var_16);
    var_25 = wp::cross(var_23, var_24);
    // face_normal_length_sq = wp.length_sq(face_normal)                                      <L 10>
    var_26 = wp::length_sq(var_25);
    // if face_normal_length_sq < 1.0e-20:                                                    <L 11>
    var_28 = (var_26 < var_27);
    if (var_28) {
        // return wp.vec3(0.0)                                                                <L 12>
        var_30 = wp::vec_t<3, wp::float32>(var_29);
        return var_30;
    }
    // face_normal /= wp.sqrt(face_normal_length_sq)                                          <L 14>
    var_31 = wp::sqrt(var_26);
    var_32 = wp::div(var_25, var_31);
    // projection = closest_point_on_triangle(center_b_world, tri_a, tri_b, tri_c)            <L 15>
    var_33 = closest_point_on_triangle_0(var_center_b_world, var_16, var_18, var_21);
    // to_centroid = (tri_a + tri_b + tri_c) / 3.0 - projection                               <L 16>
    var_34 = wp::add(var_16, var_18);
    var_35 = wp::add(var_34, var_21);
    var_37 = wp::div(var_35, var_36);
    var_38 = wp::sub(var_37, var_33);
    // to_centroid -= wp.dot(to_centroid, face_normal) * face_normal                          <L 17>
    var_39 = wp::dot(var_38, var_32);
    var_40 = wp::mul(var_39, var_32);
    var_41 = wp::sub(var_38, var_40);
    // to_centroid_length_sq = wp.length_sq(to_centroid)                                      <L 18>
    var_42 = wp::length_sq(var_41);
    // fallback_direction = -face_normal                                                      <L 20>
    var_43 = wp::neg(var_32);
    // if wp.dot(center_b_world - projection, face_normal) < 0.0:                             <L 21>
    var_44 = wp::sub(var_center_b_world, var_33);
    var_45 = wp::dot(var_44, var_32);
    var_47 = (var_45 < var_46);
    if (var_47) {
        // fallback_direction = face_normal                                                   <L 22>
        var_48 = wp::copy(var_32);
    }
    var_49 = wp::where(var_47, var_48, var_43);
    // if to_centroid_length_sq > 1.0e-20:                                                    <L 23>
    var_51 = (var_42 > var_50);
    if (var_51) {
        // fallback_direction += 0.01 * to_centroid / wp.sqrt(to_centroid_length_sq)          <L 24>
        var_53 = wp::mul(var_52, var_41);
        var_54 = wp::sqrt(var_42);
        var_55 = wp::div(var_53, var_54);
        var_56 = wp::add(var_49, var_55);
    }
    var_57 = wp::where(var_51, var_56, var_49);
    // return wp.normalize(fallback_direction) * 1.0e-5                                       <L 25>
    var_58 = wp::normalize(var_57);
    var_60 = wp::mul(var_58, var_59);
    return var_60;
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
static CUDA_CALLABLE wp::vec_t<3, wp::float32> support_map_lean_0(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    wp::int32* var_4;
    const wp::int32 var_5 = 10;
    bool var_6;
    wp::int32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::int32* var_9;
    const wp::int32 var_10 = 7;
    bool var_11;
    wp::int32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::int32* var_14;
    const wp::int32 var_15 = 3;
    bool var_16;
    wp::int32 var_17;
    wp::vec_t<3, wp::float32>* var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 1e-12;
    bool var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    const wp::float32 var_27 = 1.0;
    const wp::float32 var_28 = 0.0;
    const wp::float32 var_29 = 0.0;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    //---------
    // forward
    // def support_map_lean(geom: GenericShapeData, direction: wp.vec3, data_provider: Any) -> wp.vec3:       <L 1>
    // result = wp.vec3(0.0, 0.0, 0.0)                                                        <L 9>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // if geom.shape_type == GeoType.CONVEX_MESH:                                             <L 11>
    var_4 = &((var_geom).shape_type);
    var_7 = wp::load(var_4);
    var_6 = (var_7 == var_5);
    if (var_6) {
        // result = _support_map_convex_mesh_exhaustive(geom, direction)                      <L 12>
        var_8 = _support_map_convex_mesh_exhaustive_0(var_geom, var_direction);
    }
    if (!var_6) {
        // elif geom.shape_type == GeoType.BOX:                                               <L 14>
        var_9 = &((var_geom).shape_type);
        var_12 = wp::load(var_9);
        var_11 = (var_12 == var_10);
        if (var_11) {
            // result = _support_map_box(geom, direction)                                     <L 15>
            var_13 = _support_map_box_0(var_geom, var_direction);
        }
        if (!var_11) {
            // elif geom.shape_type == GeoType.SPHERE:                                        <L 17>
            var_14 = &((var_geom).shape_type);
            var_17 = wp::load(var_14);
            var_16 = (var_17 == var_15);
            if (var_16) {
                // radius = geom.scale[0]                                                     <L 18>
                var_18 = &((var_geom).scale);
                var_21 = wp::load(var_18);
                var_20 = wp::extract(var_21, var_19);
                // dir_len_sq = wp.length_sq(direction)                                       <L 19>
                var_22 = wp::length_sq(var_direction);
                // if dir_len_sq > 1.0e-12:                                                   <L 20>
                var_24 = (var_22 > var_23);
                if (var_24) {
                    // n = direction * _support_rsqrt_rn(dir_len_sq)                          <L 21>
                    var_25 = _support_rsqrt_rn_0(var_22);
                    var_26 = wp::mul(var_direction, var_25);
                }
                if (!var_24) {
                    // n = wp.vec3(1.0, 0.0, 0.0)                                             <L 23>
                    var_30 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
                }
                var_31 = wp::where(var_24, var_26, var_30);
                // result = n * radius                                                        <L 24>
                var_32 = wp::mul(var_31, var_20);
            }
            var_33 = wp::where(var_16, var_32, var_3);
        }
        var_34 = wp::where(var_11, var_13, var_33);
    }
    var_35 = wp::where(var_6, var_8, var_34);
    // return result                                                                          <L 26>
    return var_35;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> create_shape_support_function__locals__shape_support_26(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const bool var_4 = true;
    wp::int32* var_5;
    const wp::int32 var_6 = 7;
    bool var_7;
    wp::int32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 2;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32>* var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    const wp::float32 var_23 = 1e-06;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 2;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    bool var_35;
    const wp::float32 var_36 = 0.0;
    const wp::int32 var_37 = 0;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    bool var_40;
    const wp::float32 var_41 = 0.0;
    const wp::int32 var_42 = 1;
    const wp::int32 var_43 = 2;
    wp::float32 var_44;
    bool var_45;
    const wp::float32 var_46 = 0.0;
    const wp::int32 var_47 = 2;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    //---------
    // forward
    // def shape_support(geom: Any, direction: wp.vec3, data_provider: Any) -> wp.vec3:       <L 1>
    // result = wp.vec3(0.0, 0.0, 0.0)                                                        <L 2>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // if wp.static(fuse_builtin_box_support):                                                <L 3>
    // if geom.shape_type == GeoType.BOX:                                                     <L 4>
    var_5 = &((var_geom).shape_type);
    var_8 = wp::load(var_5);
    var_7 = (var_8 == var_6);
    if (var_7) {
        // abs_direction = wp.vec3(wp.abs(direction[0]), wp.abs(direction[1]), wp.abs(direction[2]))       <L 5>
        var_10 = wp::extract(var_direction, var_9);
        var_11 = wp::abs(var_10);
        var_13 = wp::extract(var_direction, var_12);
        var_14 = wp::abs(var_13);
        var_16 = wp::extract(var_direction, var_15);
        var_17 = wp::abs(var_16);
        var_18 = wp::vec_t<3, wp::float32>(var_11, var_14, var_17);
        // result = _support_map_box(geom, direction)                                         <L 6>
        var_19 = _support_map_box_0(var_geom, var_direction);
        // contribution = wp.cw_mul(abs_direction, geom.scale)                                <L 7>
        var_20 = &((var_geom).scale);
        var_22 = wp::load(var_20);
        var_21 = wp::cw_mul(var_18, var_22);
        // threshold = _CENTERED_BOX_SUPPORT_TIE_EPSILON * (                                  <L 8>
        // contribution[0] + contribution[1] + contribution[2]                                <L 9>
        var_25 = wp::extract(var_21, var_24);
        var_27 = wp::extract(var_21, var_26);
        var_28 = wp::add(var_25, var_27);
        var_30 = wp::extract(var_21, var_29);
        var_31 = wp::add(var_28, var_30);
        var_32 = wp::mul(var_23, var_31);
        // if contribution[0] <= threshold:                                                   <L 11>
        var_34 = wp::extract(var_21, var_33);
        var_35 = (var_34 <= var_32);
        if (var_35) {
            // result[0] = 0.0                                                                <L 12>
            wp::assign_inplace(var_19, var_37, var_36);
        }
        // if contribution[1] <= threshold:                                                   <L 13>
        var_39 = wp::extract(var_21, var_38);
        var_40 = (var_39 <= var_32);
        if (var_40) {
            // result[1] = 0.0                                                                <L 14>
            wp::assign_inplace(var_19, var_42, var_41);
        }
        // if contribution[2] <= threshold:                                                   <L 15>
        var_44 = wp::extract(var_21, var_43);
        var_45 = (var_44 <= var_32);
        if (var_45) {
            // result[2] = 0.0                                                                <L 16>
            wp::assign_inplace(var_19, var_47, var_46);
        }
    }
    if (!var_7) {
        // result = support_func(geom, direction, data_provider)                              <L 18>
        var_48 = support_map_lean_0(var_geom, var_direction, var_data_provider);
    }
    var_49 = wp::where(var_7, var_19, var_48);
    // return result                                                                          <L 32>
    return var_49;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> create_shape_support_function__locals__shape_support_27(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const bool var_4 = false;
    wp::vec_t<3, wp::float32> var_5;
    //---------
    // forward
    // def shape_support(geom: Any, direction: wp.vec3, data_provider: Any) -> wp.vec3:       <L 1>
    // result = wp.vec3(0.0, 0.0, 0.0)                                                        <L 2>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // if wp.static(fuse_builtin_box_support):                                                <L 3>
    // result = support_func(geom, direction, data_provider)                                  <L 9>
    var_5 = create_shape_support_function__locals__shape_support_26(var_geom, var_direction, var_data_provider);
    // return result                                                                          <L 10>
    return var_5;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> create_support_map_function__locals__support_map_b_17(
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_direction,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def support_map_b(                                                                     <L 1>
    // tmp = wp.quat_rotate_inv(orientation_b, direction)                                     <L 22>
    var_0 = wp::quat_rotate_inv(var_orientation_b, var_direction);
    // result = shape_support(geom_b, tmp, data_provider)                                     <L 25>
    var_1 = create_shape_support_function__locals__shape_support_27(var_geom_b, var_0, var_data_provider);
    // result = wp.quat_rotate(orientation_b, result)                                         <L 28>
    var_2 = wp::quat_rotate(var_orientation_b, var_1);
    // result = result + position_b                                                           <L 29>
    var_3 = wp::add(var_2, var_position_b);
    // return result                                                                          <L 31>
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE Vert_99873389 create_support_map_function__locals__minkowski_support_17(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_direction,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::float32 var_extend,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    Vert_99873389 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    bool var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::float32 var_8 = 0.5;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32>* var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32>* var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    //---------
    // forward
    // def minkowski_support(                                                                 <L 1>
    // v = Vert()                                                                             <L 25>
    var_0 = Vert_99873389();
    // point_a = shape_support(geom_a, direction, data_provider)                              <L 28>
    var_1 = create_shape_support_function__locals__shape_support_27(var_geom_a, var_direction, var_data_provider);
    // tmp_direction = -direction                                                             <L 31>
    var_2 = wp::neg(var_direction);
    // v.B = support_map_b(geom_b, tmp_direction, orientation_b, position_b, data_provider)       <L 32>
    var_3 = create_support_map_function__locals__support_map_b_17(var_geom_b, var_2, var_orientation_b, var_position_b, var_data_provider);
    var_0.B = var_3;
    // if extend != 0.0:                                                                      <L 35>
    var_5 = (var_extend != var_4);
    if (var_5) {
        // d = wp.normalize(direction) * extend * 0.5                                         <L 36>
        var_6 = wp::normalize(var_direction);
        var_7 = wp::mul(var_6, var_extend);
        var_9 = wp::mul(var_7, var_8);
        // point_a = point_a + d                                                              <L 37>
        var_10 = wp::add(var_1, var_9);
        // v.B = v.B - d                                                                      <L 38>
        var_11 = &((var_0).B);
        var_13 = wp::load(var_11);
        var_12 = wp::sub(var_13, var_9);
        var_0.B = var_12;
    }
    var_14 = wp::where(var_5, var_10, var_1);
    // v.BtoA = point_a - v.B                                                                 <L 41>
    var_15 = &((var_0).B);
    var_17 = wp::load(var_15);
    var_16 = wp::sub(var_14, var_17);
    var_0.BtoA = var_16;
    // return v                                                                               <L 43>
    return var_0;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:52
static CUDA_CALLABLE wp::vec_t<3, wp::float32> vert_a_0(
    Vert_99873389 var_vert)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def vert_a(vert: Vert) -> wp.vec3:                                                     <L 53>
    // return vert.B + vert.BtoA                                                              <L 55>
    var_0 = &((var_vert).B);
    var_1 = &((var_vert).BtoA);
    var_3 = wp::load(var_0);
    var_4 = wp::load(var_1);
    var_2 = wp::add(var_3, var_4);
    return var_2;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE void create_solve_mpr__locals__solve_mpr_core_8(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::float32 var_extend,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    wp::int32 var_MAX_ITER,
    wp::float32 var_COLLIDE_EPSILON,
    bool & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::float32 & ret_4)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-16;
    const wp::float32 var_1 = 0.0;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    const wp::float32 var_9 = 0.0;
    wp::vec_t<3, wp::float32> var_10;
    const wp::float32 var_11 = 0.0;
    const wp::float32 var_12 = 0.0;
    const wp::float32 var_13 = 0.0;
    wp::vec_t<3, wp::float32> var_14;
    MinkowskiCenter_f7efefbe var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::float32 var_19;
    bool var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    bool var_27;
    const wp::float32 var_28 = -1e+30;
    wp::float32 var_29;
    const wp::float32 var_30 = 1.0;
    const wp::float32 var_31 = 0.0;
    const wp::float32 var_32 = 0.0;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 0;
    const wp::float32 var_35 = 0.0;
    const wp::float32 var_36 = 0.0;
    const wp::float32 var_37 = 0.0;
    wp::vec_t<3, wp::float32> var_38;
    const wp::float32 var_39 = 1.0;
    Vert_99873389 var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    bool var_44;
    wp::float32 var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::float32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    const wp::int32 var_49 = 1;
    const wp::float32 var_50 = 0.0;
    const wp::float32 var_51 = 0.0;
    const wp::float32 var_52 = 0.0;
    wp::vec_t<3, wp::float32> var_53;
    const wp::float32 var_54 = 1.0;
    Vert_99873389 var_55;
    wp::vec_t<3, wp::float32>* var_56;
    wp::float32 var_57;
    wp::vec_t<3, wp::float32> var_58;
    bool var_59;
    wp::float32 var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    const wp::int32 var_64 = 2;
    const wp::float32 var_65 = 0.0;
    const wp::float32 var_66 = 0.0;
    const wp::float32 var_67 = 0.0;
    wp::vec_t<3, wp::float32> var_68;
    const wp::float32 var_69 = 1.0;
    Vert_99873389 var_70;
    wp::vec_t<3, wp::float32>* var_71;
    wp::float32 var_72;
    wp::vec_t<3, wp::float32> var_73;
    bool var_74;
    wp::float32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::float32 var_77;
    wp::vec_t<3, wp::float32> var_78;
    const wp::float32 var_79 = 1e-05;
    wp::vec_t<3, wp::float32> var_80;
    wp::vec_t<3, wp::float32>* var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    Vert_99873389 var_84;
    wp::vec_t<3, wp::float32> var_85;
    wp::vec_t<3, wp::float32>* var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32>* var_89;
    wp::float32 var_90;
    wp::vec_t<3, wp::float32> var_91;
    const wp::float32 var_92 = 0.0;
    bool var_93;
    const bool var_94 = false;
    wp::vec_t<3, wp::float32>* var_95;
    wp::vec_t<3, wp::float32>* var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    bool var_102;
    wp::vec_t<3, wp::float32>* var_103;
    wp::vec_t<3, wp::float32>* var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32>* var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::float32 var_112;
    const bool var_113 = true;
    wp::float32 var_114;
    wp::vec_t<3, wp::float32> var_115;
    Vert_99873389 var_116;
    wp::vec_t<3, wp::float32>* var_117;
    wp::float32 var_118;
    wp::vec_t<3, wp::float32> var_119;
    const wp::float32 var_120 = 0.0;
    bool var_121;
    const bool var_122 = false;
    wp::vec_t<3, wp::float32>* var_123;
    wp::vec_t<3, wp::float32>* var_124;
    wp::vec_t<3, wp::float32> var_125;
    wp::vec_t<3, wp::float32> var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::vec_t<3, wp::float32>* var_128;
    wp::vec_t<3, wp::float32>* var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::vec_t<3, wp::float32> var_131;
    wp::vec_t<3, wp::float32> var_132;
    wp::vec_t<3, wp::float32> var_133;
    wp::vec_t<3, wp::float32>* var_134;
    wp::float32 var_135;
    wp::vec_t<3, wp::float32> var_136;
    const wp::float32 var_137 = 0.0;
    bool var_138;
    wp::vec_t<3, wp::float32>* var_139;
    wp::vec_t<3, wp::float32> var_140;
    wp::vec_t<3, wp::float32> var_141;
    wp::vec_t<3, wp::float32>* var_142;
    wp::vec_t<3, wp::float32> var_143;
    wp::vec_t<3, wp::float32> var_144;
    wp::vec_t<3, wp::float32>* var_145;
    wp::vec_t<3, wp::float32> var_146;
    wp::vec_t<3, wp::float32> var_147;
    wp::vec_t<3, wp::float32>* var_148;
    wp::vec_t<3, wp::float32> var_149;
    wp::vec_t<3, wp::float32> var_150;
    wp::vec_t<3, wp::float32> var_151;
    wp::vec_t<3, wp::float32> var_152;
    const wp::int32 var_153 = 0;
    wp::int32 var_154;
    const wp::int32 var_155 = 0;
    wp::int32 var_156;
    const bool var_157 = false;
    bool var_158;
    Vert_99873389 var_159;
    const bool var_160 = true;
    bool var_161;
    const bool var_162 = false;
    const wp::int32 var_163 = 1;
    wp::int32 var_164;
    Vert_99873389 var_165;
    wp::vec_t<3, wp::float32>* var_166;
    wp::float32 var_167;
    wp::vec_t<3, wp::float32> var_168;
    const wp::float32 var_169 = 0.0;
    bool var_170;
    const bool var_171 = false;
    wp::vec_t<3, wp::float32>* var_172;
    wp::vec_t<3, wp::float32>* var_173;
    wp::vec_t<3, wp::float32> var_174;
    wp::vec_t<3, wp::float32> var_175;
    wp::vec_t<3, wp::float32> var_176;
    wp::vec_t<3, wp::float32>* var_177;
    wp::float32 var_178;
    wp::vec_t<3, wp::float32> var_179;
    const wp::float32 var_180 = 0.0;
    bool var_181;
    Vert_99873389 var_182;
    wp::vec_t<3, wp::float32>* var_183;
    wp::vec_t<3, wp::float32>* var_184;
    wp::vec_t<3, wp::float32> var_185;
    wp::vec_t<3, wp::float32> var_186;
    wp::vec_t<3, wp::float32> var_187;
    wp::vec_t<3, wp::float32>* var_188;
    wp::vec_t<3, wp::float32>* var_189;
    wp::vec_t<3, wp::float32> var_190;
    wp::vec_t<3, wp::float32> var_191;
    wp::vec_t<3, wp::float32> var_192;
    wp::vec_t<3, wp::float32> var_193;
    wp::vec_t<3, wp::float32> var_194;
    wp::int32 var_195;
    Vert_99873389 var_196;
    wp::vec_t<3, wp::float32>* var_197;
    wp::vec_t<3, wp::float32>* var_198;
    wp::vec_t<3, wp::float32> var_199;
    wp::vec_t<3, wp::float32> var_200;
    wp::vec_t<3, wp::float32> var_201;
    wp::vec_t<3, wp::float32>* var_202;
    wp::float32 var_203;
    wp::vec_t<3, wp::float32> var_204;
    const wp::float32 var_205 = 0.0;
    bool var_206;
    Vert_99873389 var_207;
    wp::vec_t<3, wp::float32>* var_208;
    wp::vec_t<3, wp::float32>* var_209;
    wp::vec_t<3, wp::float32> var_210;
    wp::vec_t<3, wp::float32> var_211;
    wp::vec_t<3, wp::float32> var_212;
    wp::vec_t<3, wp::float32>* var_213;
    wp::vec_t<3, wp::float32>* var_214;
    wp::vec_t<3, wp::float32> var_215;
    wp::vec_t<3, wp::float32> var_216;
    wp::vec_t<3, wp::float32> var_217;
    wp::vec_t<3, wp::float32> var_218;
    wp::vec_t<3, wp::float32> var_219;
    wp::int32 var_220;
    Vert_99873389 var_221;
    Vert_99873389 var_222;
    const bool var_223 = true;
    const wp::int32 var_224 = 1;
    wp::int32 var_225;
    wp::vec_t<3, wp::float32>* var_226;
    wp::vec_t<3, wp::float32>* var_227;
    wp::vec_t<3, wp::float32> var_228;
    wp::vec_t<3, wp::float32> var_229;
    wp::vec_t<3, wp::float32> var_230;
    wp::vec_t<3, wp::float32>* var_231;
    wp::vec_t<3, wp::float32>* var_232;
    wp::vec_t<3, wp::float32> var_233;
    wp::vec_t<3, wp::float32> var_234;
    wp::vec_t<3, wp::float32> var_235;
    wp::vec_t<3, wp::float32> var_236;
    wp::float32 var_237;
    wp::float32 var_238;
    bool var_239;
    const bool var_240 = false;
    bool var_241;
    wp::vec_t<3, wp::float32>* var_242;
    wp::float32 var_243;
    wp::vec_t<3, wp::float32> var_244;
    const wp::float32 var_245 = 0.0;
    bool var_246;
    wp::float32 var_247;
    bool var_248;
    Vert_99873389 var_249;
    wp::vec_t<3, wp::float32>* var_250;
    wp::vec_t<3, wp::float32>* var_251;
    wp::vec_t<3, wp::float32> var_252;
    wp::vec_t<3, wp::float32> var_253;
    wp::vec_t<3, wp::float32> var_254;
    wp::float32 var_255;
    wp::vec_t<3, wp::float32>* var_256;
    wp::float32 var_257;
    wp::vec_t<3, wp::float32> var_258;
    bool var_259;
    wp::float32 var_260;
    wp::float32 var_261;
    wp::float32 var_262;
    bool var_263;
    const wp::float32 var_264 = 0.0;
    bool var_265;
    bool var_266;
    const wp::float32 var_267 = 1.0;
    wp::float32 var_268;
    wp::float32 var_269;
    wp::float32 var_270;
    wp::vec_t<3, wp::float32> var_271;
    wp::vec_t<3, wp::float32>* var_272;
    wp::vec_t<3, wp::float32> var_273;
    wp::vec_t<3, wp::float32> var_274;
    wp::float32 var_275;
    wp::float32 var_276;
    wp::vec_t<3, wp::float32>* var_277;
    wp::vec_t<3, wp::float32> var_278;
    wp::vec_t<3, wp::float32> var_279;
    wp::float32 var_280;
    wp::float32 var_281;
    const wp::float32 var_282 = 1.0;
    wp::float32 var_283;
    wp::float32 var_284;
    wp::vec_t<3, wp::float32> var_285;
    wp::vec_t<3, wp::float32> var_286;
    wp::vec_t<3, wp::float32> var_287;
    wp::vec_t<3, wp::float32> var_288;
    wp::vec_t<3, wp::float32> var_289;
    wp::vec_t<3, wp::float32> var_290;
    wp::vec_t<3, wp::float32> var_291;
    wp::vec_t<3, wp::float32> var_292;
    wp::vec_t<3, wp::float32>* var_293;
    wp::vec_t<3, wp::float32> var_294;
    wp::vec_t<3, wp::float32> var_295;
    wp::vec_t<3, wp::float32>* var_296;
    wp::vec_t<3, wp::float32> var_297;
    wp::vec_t<3, wp::float32> var_298;
    wp::vec_t<3, wp::float32> var_299;
    wp::vec_t<3, wp::float32>* var_300;
    wp::vec_t<3, wp::float32> var_301;
    wp::vec_t<3, wp::float32> var_302;
    wp::vec_t<3, wp::float32> var_303;
    wp::float32 var_304;
    wp::vec_t<3, wp::float32> var_305;
    wp::vec_t<3, wp::float32> var_306;
    wp::vec_t<3, wp::float32> var_307;
    wp::vec_t<3, wp::float32> var_308;
    wp::float32 var_309;
    wp::vec_t<3, wp::float32> var_310;
    wp::vec_t<3, wp::float32> var_311;
    wp::vec_t<3, wp::float32> var_312;
    wp::vec_t<3, wp::float32> var_313;
    wp::vec_t<3, wp::float32>* var_314;
    wp::vec_t<3, wp::float32>* var_315;
    wp::vec_t<3, wp::float32> var_316;
    wp::vec_t<3, wp::float32> var_317;
    wp::vec_t<3, wp::float32> var_318;
    wp::vec_t<3, wp::float32>* var_319;
    wp::float32 var_320;
    wp::vec_t<3, wp::float32> var_321;
    const wp::float32 var_322 = 0.0;
    bool var_323;
    wp::vec_t<3, wp::float32>* var_324;
    wp::float32 var_325;
    wp::vec_t<3, wp::float32> var_326;
    const wp::float32 var_327 = 0.0;
    bool var_328;
    Vert_99873389 var_329;
    Vert_99873389 var_330;
    Vert_99873389 var_331;
    Vert_99873389 var_332;
    wp::vec_t<3, wp::float32>* var_333;
    wp::float32 var_334;
    wp::vec_t<3, wp::float32> var_335;
    const wp::float32 var_336 = 0.0;
    bool var_337;
    Vert_99873389 var_338;
    Vert_99873389 var_339;
    Vert_99873389 var_340;
    Vert_99873389 var_341;
    Vert_99873389 var_342;
    Vert_99873389 var_343;
    Vert_99873389 var_344;
    wp::float32 var_345;
    //---------
    // forward
    // def solve_mpr_core(                                                                    <L 1>
    // NUMERIC_EPSILON = 1e-16                                                                <L 37>
    // penetration = float(0.0)                                                               <L 40>
    var_2 = wp::float(var_1);
    // point_a = wp.vec3(0.0, 0.0, 0.0)                                                       <L 41>
    var_6 = wp::vec_t<3, wp::float32>(var_3, var_4, var_5);
    // point_b = wp.vec3(0.0, 0.0, 0.0)                                                       <L 42>
    var_10 = wp::vec_t<3, wp::float32>(var_7, var_8, var_9);
    // normal = wp.vec3(0.0, 0.0, 0.0)                                                        <L 43>
    var_14 = wp::vec_t<3, wp::float32>(var_11, var_12, var_13);
    // v0 = geometric_center(geom_a, geom_b, orientation_b, position_b, data_provider)        <L 46>
    var_15 = create_shape_center_function__locals__shape_center_16(var_geom_a, var_geom_b, var_orientation_b, var_position_b, var_data_provider);
    // normal = v0.BtoA                                                                       <L 48>
    var_16 = &((var_15).BtoA);
    var_18 = wp::load(var_16);
    var_17 = wp::copy(var_18);
    // if wp.length_sq(normal) < NUMERIC_EPSILON:                                             <L 49>
    var_19 = wp::length_sq(var_17);
    var_20 = (var_19 < var_0);
    if (var_20) {
        // v0.BtoA = geometric_center.fallback(geom_a, v0.B)                                  <L 50>
        var_21 = &((var_15).B);
        var_23 = wp::load(var_21);
        var_22 = _minkowski_center_fallback_0(var_geom_a, var_23);
        var_15.BtoA = var_22;
        // if wp.length_sq(v0.BtoA) < NUMERIC_EPSILON:                                        <L 51>
        var_24 = &((var_15).BtoA);
        var_26 = wp::load(var_24);
        var_25 = wp::length_sq(var_26);
        var_27 = (var_25 < var_0);
        if (var_27) {
            // best_dot = float(-1.0e30)                                                      <L 53>
            var_29 = wp::float(var_28);
            // best_dir = wp.vec3(1.0, 0.0, 0.0)                                              <L 54>
            var_33 = wp::vec_t<3, wp::float32>(var_30, var_31, var_32);
            // for axis_idx in range(3):                                                      <L 55>
            // probe = wp.vec3(0.0, 0.0, 0.0)                                                 <L 56>
            var_38 = wp::vec_t<3, wp::float32>(var_35, var_36, var_37);
            // probe[axis_idx] = 1.0                                                          <L 57>
            wp::assign_inplace(var_38, var_34, var_39);
            // sv = mpr_support(geom_a, geom_b, probe, orientation_b, position_b, extend, data_provider)       <L 58>
            var_40 = create_support_map_function__locals__minkowski_support_17(var_geom_a, var_geom_b, var_38, var_orientation_b, var_position_b, var_extend, var_data_provider);
            // d = wp.dot(sv.BtoA, probe)                                                     <L 59>
            var_41 = &((var_40).BtoA);
            var_43 = wp::load(var_41);
            var_42 = wp::dot(var_43, var_38);
            // if d > best_dot:                                                               <L 60>
            var_44 = (var_42 > var_29);
            if (var_44) {
                // best_dot = d                                                               <L 61>
                var_45 = wp::copy(var_42);
                // best_dir = probe                                                           <L 62>
                var_46 = wp::copy(var_38);
            }
            var_47 = wp::where(var_44, var_45, var_29);
            var_48 = wp::where(var_44, var_46, var_33);
            // probe = wp.vec3(0.0, 0.0, 0.0)                                                 <L 56>
            var_53 = wp::vec_t<3, wp::float32>(var_50, var_51, var_52);
            // probe[axis_idx] = 1.0                                                          <L 57>
            wp::assign_inplace(var_53, var_49, var_54);
            // sv = mpr_support(geom_a, geom_b, probe, orientation_b, position_b, extend, data_provider)       <L 58>
            var_55 = create_support_map_function__locals__minkowski_support_17(var_geom_a, var_geom_b, var_53, var_orientation_b, var_position_b, var_extend, var_data_provider);
            // d = wp.dot(sv.BtoA, probe)                                                     <L 59>
            var_56 = &((var_55).BtoA);
            var_58 = wp::load(var_56);
            var_57 = wp::dot(var_58, var_53);
            // if d > best_dot:                                                               <L 60>
            var_59 = (var_57 > var_47);
            if (var_59) {
                // best_dot = d                                                               <L 61>
                var_60 = wp::copy(var_57);
                // best_dir = probe                                                           <L 62>
                var_61 = wp::copy(var_53);
            }
            var_62 = wp::where(var_59, var_60, var_47);
            var_63 = wp::where(var_59, var_61, var_48);
            // probe = wp.vec3(0.0, 0.0, 0.0)                                                 <L 56>
            var_68 = wp::vec_t<3, wp::float32>(var_65, var_66, var_67);
            // probe[axis_idx] = 1.0                                                          <L 57>
            wp::assign_inplace(var_68, var_64, var_69);
            // sv = mpr_support(geom_a, geom_b, probe, orientation_b, position_b, extend, data_provider)       <L 58>
            var_70 = create_support_map_function__locals__minkowski_support_17(var_geom_a, var_geom_b, var_68, var_orientation_b, var_position_b, var_extend, var_data_provider);
            // d = wp.dot(sv.BtoA, probe)                                                     <L 59>
            var_71 = &((var_70).BtoA);
            var_73 = wp::load(var_71);
            var_72 = wp::dot(var_73, var_68);
            // if d > best_dot:                                                               <L 60>
            var_74 = (var_72 > var_62);
            if (var_74) {
                // best_dot = d                                                               <L 61>
                var_75 = wp::copy(var_72);
                // best_dir = probe                                                           <L 62>
                var_76 = wp::copy(var_68);
            }
            var_77 = wp::where(var_74, var_75, var_62);
            var_78 = wp::where(var_74, var_76, var_63);
            // v0.BtoA = best_dir * 1e-05                                                     <L 63>
            var_80 = wp::mul(var_78, var_79);
            var_15.BtoA = var_80;
        }
    }
    // normal = -v0.BtoA                                                                      <L 65>
    var_81 = &((var_15).BtoA);
    var_83 = wp::load(var_81);
    var_82 = wp::neg(var_83);
    // v1 = mpr_support(geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider)       <L 68>
    var_84 = create_support_map_function__locals__minkowski_support_17(var_geom_a, var_geom_b, var_82, var_orientation_b, var_position_b, var_extend, var_data_provider);
    // point_a = vert_a(v1)                                                                   <L 70>
    var_85 = vert_a_0(var_84);
    // point_b = v1.B                                                                         <L 71>
    var_86 = &((var_84).B);
    var_88 = wp::load(var_86);
    var_87 = wp::copy(var_88);
    // if wp.dot(v1.BtoA, normal) <= 0.0:                                                     <L 73>
    var_89 = &((var_84).BtoA);
    var_91 = wp::load(var_89);
    var_90 = wp::dot(var_91, var_82);
    var_93 = (var_90 <= var_92);
    if (var_93) {
        // return False, point_a, point_b, normal, penetration                                <L 74>
        ret_0 = var_94;
        ret_1 = var_85;
        ret_2 = var_87;
        ret_3 = var_82;
        ret_4 = var_2;
        return;
    }
    // normal = wp.cross(v1.BtoA, v0.BtoA)                                                    <L 76>
    var_95 = &((var_84).BtoA);
    var_96 = &((var_15).BtoA);
    var_98 = wp::load(var_95);
    var_99 = wp::load(var_96);
    var_97 = wp::cross(var_98, var_99);
    // if wp.length_sq(normal) < NUMERIC_EPSILON * NUMERIC_EPSILON:                           <L 78>
    var_100 = wp::length_sq(var_97);
    var_101 = wp::mul(var_0, var_0);
    var_102 = (var_100 < var_101);
    if (var_102) {
        // normal = v1.BtoA - v0.BtoA                                                         <L 79>
        var_103 = &((var_84).BtoA);
        var_104 = &((var_15).BtoA);
        var_106 = wp::load(var_103);
        var_107 = wp::load(var_104);
        var_105 = wp::sub(var_106, var_107);
        // normal = wp.normalize(normal)                                                      <L 80>
        var_108 = wp::normalize(var_105);
        // temp1 = v1.BtoA                                                                    <L 82>
        var_109 = &((var_84).BtoA);
        var_111 = wp::load(var_109);
        var_110 = wp::copy(var_111);
        // penetration = wp.dot(temp1, normal)                                                <L 83>
        var_112 = wp::dot(var_110, var_108);
        // return True, point_a, point_b, normal, penetration                                 <L 85>
        ret_0 = var_113;
        ret_1 = var_85;
        ret_2 = var_87;
        ret_3 = var_108;
        ret_4 = var_112;
        return;
    }
    var_114 = wp::where(var_102, var_112, var_2);
    var_115 = wp::where(var_102, var_108, var_97);
    // v2 = mpr_support(geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider)       <L 88>
    var_116 = create_support_map_function__locals__minkowski_support_17(var_geom_a, var_geom_b, var_115, var_orientation_b, var_position_b, var_extend, var_data_provider);
    // if wp.dot(v2.BtoA, normal) <= 0.0:                                                     <L 90>
    var_117 = &((var_116).BtoA);
    var_119 = wp::load(var_117);
    var_118 = wp::dot(var_119, var_115);
    var_121 = (var_118 <= var_120);
    if (var_121) {
        // return False, point_a, point_b, normal, penetration                                <L 91>
        ret_0 = var_122;
        ret_1 = var_85;
        ret_2 = var_87;
        ret_3 = var_115;
        ret_4 = var_114;
        return;
    }
    // temp1 = v1.BtoA - v0.BtoA                                                              <L 94>
    var_123 = &((var_84).BtoA);
    var_124 = &((var_15).BtoA);
    var_126 = wp::load(var_123);
    var_127 = wp::load(var_124);
    var_125 = wp::sub(var_126, var_127);
    // temp2 = v2.BtoA - v0.BtoA                                                              <L 95>
    var_128 = &((var_116).BtoA);
    var_129 = &((var_15).BtoA);
    var_131 = wp::load(var_128);
    var_132 = wp::load(var_129);
    var_130 = wp::sub(var_131, var_132);
    // normal = wp.cross(temp1, temp2)                                                        <L 96>
    var_133 = wp::cross(var_125, var_130);
    // dist = wp.dot(normal, v0.BtoA)                                                         <L 98>
    var_134 = &((var_15).BtoA);
    var_136 = wp::load(var_134);
    var_135 = wp::dot(var_133, var_136);
    // if dist > 0.0:                                                                         <L 101>
    var_138 = (var_135 > var_137);
    if (var_138) {
        // tmp_b = v1.B                                                                       <L 103>
        var_139 = &((var_84).B);
        var_141 = wp::load(var_139);
        var_140 = wp::copy(var_141);
        // tmp_btoa = v1.BtoA                                                                 <L 104>
        var_142 = &((var_84).BtoA);
        var_144 = wp::load(var_142);
        var_143 = wp::copy(var_144);
        // v1.B = v2.B                                                                        <L 105>
        var_145 = &((var_116).B);
        var_147 = wp::load(var_145);
        var_146 = wp::copy(var_147);
        var_84.B = var_146;
        // v1.BtoA = v2.BtoA                                                                  <L 106>
        var_148 = &((var_116).BtoA);
        var_150 = wp::load(var_148);
        var_149 = wp::copy(var_150);
        var_84.BtoA = var_149;
        // v2.B = tmp_b                                                                       <L 107>
        var_116.B = var_140;
        // v2.BtoA = tmp_btoa                                                                 <L 108>
        var_116.BtoA = var_143;
        // normal = -normal                                                                   <L 109>
        var_151 = wp::neg(var_133);
    }
    var_152 = wp::where(var_138, var_151, var_133);
    // phase1 = int(0)                                                                        <L 111>
    var_154 = wp::int(var_153);
    // phase2 = int(0)                                                                        <L 112>
    var_156 = wp::int(var_155);
    // hit = bool(False)                                                                      <L 113>
    var_158 = bool(var_157);
    // v3 = Vert()                                                                            <L 116>
    var_159 = Vert_99873389();
    // while True:                                                                            <L 117>
    start_while_3:;
    if ((var_160) == false) goto end_while_3;
        // if phase1 > MAX_ITER:                                                              <L 118>
        var_161 = (var_154 > var_MAX_ITER);
        if (var_161) {
            // return False, point_a, point_b, normal, penetration                            <L 119>
            ret_0 = var_162;
            ret_1 = var_85;
            ret_2 = var_87;
            ret_3 = var_152;
            ret_4 = var_114;
            return;
        }
        // phase1 += 1                                                                        <L 121>
        var_164 = wp::add(var_154, var_163);
        // v3 = mpr_support(geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider)       <L 123>
        var_165 = create_support_map_function__locals__minkowski_support_17(var_geom_a, var_geom_b, var_152, var_orientation_b, var_position_b, var_extend, var_data_provider);
        // if wp.dot(v3.BtoA, normal) <= 0.0:                                                 <L 125>
        var_166 = &((var_165).BtoA);
        var_168 = wp::load(var_166);
        var_167 = wp::dot(var_168, var_152);
        var_170 = (var_167 <= var_169);
        if (var_170) {
            // return False, point_a, point_b, normal, penetration                            <L 126>
            ret_0 = var_171;
            ret_1 = var_85;
            ret_2 = var_87;
            ret_3 = var_152;
            ret_4 = var_114;
            return;
        }
        // temp1 = wp.cross(v1.BtoA, v3.BtoA)                                                 <L 129>
        var_172 = &((var_84).BtoA);
        var_173 = &((var_165).BtoA);
        var_175 = wp::load(var_172);
        var_176 = wp::load(var_173);
        var_174 = wp::cross(var_175, var_176);
        // if wp.dot(temp1, v0.BtoA) < 0.0:                                                   <L 130>
        var_177 = &((var_15).BtoA);
        var_179 = wp::load(var_177);
        var_178 = wp::dot(var_174, var_179);
        var_181 = (var_178 < var_180);
        if (var_181) {
            // v2 = v3                                                                        <L 131>
            var_182 = wp::copy(var_165);
            // temp1 = v1.BtoA - v0.BtoA                                                      <L 132>
            var_183 = &((var_84).BtoA);
            var_184 = &((var_15).BtoA);
            var_186 = wp::load(var_183);
            var_187 = wp::load(var_184);
            var_185 = wp::sub(var_186, var_187);
            // temp2 = v3.BtoA - v0.BtoA                                                      <L 133>
            var_188 = &((var_165).BtoA);
            var_189 = &((var_15).BtoA);
            var_191 = wp::load(var_188);
            var_192 = wp::load(var_189);
            var_190 = wp::sub(var_191, var_192);
            // normal = wp.cross(temp1, temp2)                                                <L 134>
            var_193 = wp::cross(var_185, var_190);
            // continue                                                                       <L 135>
            wp::assign(var_152, var_193);
            wp::assign(var_125, var_185);
            wp::assign(var_116, var_182);
            wp::assign(var_130, var_190);
            wp::assign(var_154, var_164);
            wp::assign(var_159, var_165);
            goto start_while_3;
        }
        var_194 = wp::where(var_181, var_125, var_174);
        var_195 = wp::where(var_181, var_154, var_164);
        var_196 = wp::where(var_181, var_159, var_165);
        // temp1 = wp.cross(v3.BtoA, v2.BtoA)                                                 <L 138>
        var_197 = &((var_196).BtoA);
        var_198 = &((var_116).BtoA);
        var_200 = wp::load(var_197);
        var_201 = wp::load(var_198);
        var_199 = wp::cross(var_200, var_201);
        // if wp.dot(temp1, v0.BtoA) < 0.0:                                                   <L 139>
        var_202 = &((var_15).BtoA);
        var_204 = wp::load(var_202);
        var_203 = wp::dot(var_199, var_204);
        var_206 = (var_203 < var_205);
        if (var_206) {
            // v1 = v3                                                                        <L 140>
            var_207 = wp::copy(var_196);
            // temp1 = v3.BtoA - v0.BtoA                                                      <L 141>
            var_208 = &((var_196).BtoA);
            var_209 = &((var_15).BtoA);
            var_211 = wp::load(var_208);
            var_212 = wp::load(var_209);
            var_210 = wp::sub(var_211, var_212);
            // temp2 = v2.BtoA - v0.BtoA                                                      <L 142>
            var_213 = &((var_116).BtoA);
            var_214 = &((var_15).BtoA);
            var_216 = wp::load(var_213);
            var_217 = wp::load(var_214);
            var_215 = wp::sub(var_216, var_217);
            // normal = wp.cross(temp1, temp2)                                                <L 143>
            var_218 = wp::cross(var_210, var_215);
            // continue                                                                       <L 144>
            wp::assign(var_152, var_218);
            wp::assign(var_84, var_207);
            wp::assign(var_125, var_210);
            wp::assign(var_130, var_215);
            wp::assign(var_154, var_195);
            wp::assign(var_159, var_196);
            goto start_while_3;
        }
        var_219 = wp::where(var_206, var_125, var_199);
        var_220 = wp::where(var_206, var_154, var_195);
        var_221 = wp::where(var_206, var_159, var_196);
        // break                                                                              <L 146>
        wp::assign(var_125, var_219);
        wp::assign(var_154, var_220);
        wp::assign(var_159, var_221);
        goto end_while_3;
    goto start_while_3;
    end_while_3:;
    // v4 = Vert()                                                                            <L 149>
    var_222 = Vert_99873389();
    // while True:                                                                            <L 150>
    start_while_7:;
    if ((var_223) == false) goto end_while_7;
        // phase2 += 1                                                                        <L 151>
        var_225 = wp::add(var_156, var_224);
        // temp1 = v2.BtoA - v1.BtoA                                                          <L 154>
        var_226 = &((var_116).BtoA);
        var_227 = &((var_84).BtoA);
        var_229 = wp::load(var_226);
        var_230 = wp::load(var_227);
        var_228 = wp::sub(var_229, var_230);
        // temp2 = v3.BtoA - v1.BtoA                                                          <L 155>
        var_231 = &((var_159).BtoA);
        var_232 = &((var_84).BtoA);
        var_234 = wp::load(var_231);
        var_235 = wp::load(var_232);
        var_233 = wp::sub(var_234, var_235);
        // normal = wp.cross(temp1, temp2)                                                    <L 156>
        var_236 = wp::cross(var_228, var_233);
        // normal_sq = wp.length_sq(normal)                                                   <L 158>
        var_237 = wp::length_sq(var_236);
        // if normal_sq < NUMERIC_EPSILON * NUMERIC_EPSILON:                                  <L 161>
        var_238 = wp::mul(var_0, var_0);
        var_239 = (var_237 < var_238);
        if (var_239) {
            // return False, point_a, point_b, normal, penetration                            <L 162>
            ret_0 = var_240;
            ret_1 = var_85;
            ret_2 = var_87;
            ret_3 = var_236;
            ret_4 = var_114;
            return;
        }
        // if not hit:                                                                        <L 164>
        var_241 = wp::unot(var_158);
        if (var_241) {
            // d = wp.dot(normal, v1.BtoA)                                                    <L 166>
            var_242 = &((var_84).BtoA);
            var_244 = wp::load(var_242);
            var_243 = wp::dot(var_236, var_244);
            // hit = d >= 0.0                                                                 <L 168>
            var_246 = (var_243 >= var_245);
        }
        var_247 = wp::where(var_241, var_243, var_72);
        var_248 = wp::where(var_241, var_246, var_158);
        // v4 = mpr_support(geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider)       <L 170>
        var_249 = create_support_map_function__locals__minkowski_support_17(var_geom_a, var_geom_b, var_236, var_orientation_b, var_position_b, var_extend, var_data_provider);
        // temp3 = v4.BtoA - v3.BtoA                                                          <L 172>
        var_250 = &((var_249).BtoA);
        var_251 = &((var_159).BtoA);
        var_253 = wp::load(var_250);
        var_254 = wp::load(var_251);
        var_252 = wp::sub(var_253, var_254);
        // delta = wp.dot(temp3, normal)                                                      <L 173>
        var_255 = wp::dot(var_252, var_236);
        // penetration = wp.dot(v4.BtoA, normal)                                              <L 174>
        var_256 = &((var_249).BtoA);
        var_258 = wp::load(var_256);
        var_257 = wp::dot(var_258, var_236);
        // if (                                                                               <L 177>
        // delta * delta <= COLLIDE_EPSILON * COLLIDE_EPSILON * normal_sq                     <L 178>
        var_260 = wp::mul(var_255, var_255);
        var_261 = wp::mul(var_COLLIDE_EPSILON, var_COLLIDE_EPSILON);
        var_262 = wp::mul(var_261, var_237);
        var_263 = (var_260 <= var_262);
        var_259 = var_263;
        if (!var_259) {
            // or penetration <= 0.0                                                          <L 179>
            var_265 = (var_257 <= var_264);
            var_259 = var_259 || var_265;
        }
        if (!var_259) {
            // or phase2 > MAX_ITER                                                           <L 180>
            var_266 = (var_225 > var_MAX_ITER);
            var_259 = var_259 || var_266;
        }
        if (var_259) {
            // if hit:                                                                        <L 182>
            if (var_248) {
                // inv_normal = 1.0 / wp.sqrt(normal_sq)                                      <L 183>
                var_268 = wp::sqrt(var_237);
                var_269 = wp::div(var_267, var_268);
                // penetration *= inv_normal                                                  <L 184>
                var_270 = wp::mul(var_257, var_269);
                // normal = normal * inv_normal                                               <L 185>
                var_271 = wp::mul(var_236, var_269);
                // temp3 = wp.cross(v1.BtoA, temp1)                                           <L 188>
                var_272 = &((var_84).BtoA);
                var_274 = wp::load(var_272);
                var_273 = wp::cross(var_274, var_228);
                // gamma = wp.dot(temp3, normal) * inv_normal                                 <L 189>
                var_275 = wp::dot(var_273, var_271);
                var_276 = wp::mul(var_275, var_269);
                // temp3 = wp.cross(temp2, v1.BtoA)                                           <L 190>
                var_277 = &((var_84).BtoA);
                var_279 = wp::load(var_277);
                var_278 = wp::cross(var_233, var_279);
                // beta = wp.dot(temp3, normal) * inv_normal                                  <L 191>
                var_280 = wp::dot(var_278, var_271);
                var_281 = wp::mul(var_280, var_269);
                // alpha = 1.0 - gamma - beta                                                 <L 192>
                var_283 = wp::sub(var_282, var_276);
                var_284 = wp::sub(var_283, var_281);
                // point_a = alpha * vert_a(v1) + beta * vert_a(v2) + gamma * vert_a(v3)       <L 194>
                var_285 = vert_a_0(var_84);
                var_286 = wp::mul(var_284, var_285);
                var_287 = vert_a_0(var_116);
                var_288 = wp::mul(var_281, var_287);
                var_289 = wp::add(var_286, var_288);
                var_290 = vert_a_0(var_159);
                var_291 = wp::mul(var_276, var_290);
                var_292 = wp::add(var_289, var_291);
                // point_b = alpha * v1.B + beta * v2.B + gamma * v3.B                        <L 195>
                var_293 = &((var_84).B);
                var_295 = wp::load(var_293);
                var_294 = wp::mul(var_284, var_295);
                var_296 = &((var_116).B);
                var_298 = wp::load(var_296);
                var_297 = wp::mul(var_281, var_298);
                var_299 = wp::add(var_294, var_297);
                var_300 = &((var_159).B);
                var_302 = wp::load(var_300);
                var_301 = wp::mul(var_276, var_302);
                var_303 = wp::add(var_299, var_301);
            }
            var_304 = wp::where(var_248, var_270, var_257);
            var_305 = wp::where(var_248, var_292, var_85);
            var_306 = wp::where(var_248, var_303, var_87);
            var_307 = wp::where(var_248, var_271, var_236);
            var_308 = wp::where(var_248, var_278, var_252);
            // return hit, point_a, point_b, normal, penetration                              <L 197>
            ret_0 = var_248;
            ret_1 = var_305;
            ret_2 = var_306;
            ret_3 = var_307;
            ret_4 = var_304;
            return;
        }
        var_309 = wp::where(var_259, var_304, var_257);
        var_310 = wp::where(var_259, var_305, var_85);
        var_311 = wp::where(var_259, var_306, var_87);
        var_312 = wp::where(var_259, var_307, var_236);
        var_313 = wp::where(var_259, var_308, var_252);
        // temp1 = wp.cross(v4.BtoA, v0.BtoA)                                                 <L 200>
        var_314 = &((var_249).BtoA);
        var_315 = &((var_15).BtoA);
        var_317 = wp::load(var_314);
        var_318 = wp::load(var_315);
        var_316 = wp::cross(var_317, var_318);
        // dot = wp.dot(temp1, v1.BtoA)                                                       <L 201>
        var_319 = &((var_84).BtoA);
        var_321 = wp::load(var_319);
        var_320 = wp::dot(var_316, var_321);
        // if dot >= 0.0:                                                                     <L 203>
        var_323 = (var_320 >= var_322);
        if (var_323) {
            // dot = wp.dot(temp1, v2.BtoA)                                                   <L 205>
            var_324 = &((var_116).BtoA);
            var_326 = wp::load(var_324);
            var_325 = wp::dot(var_316, var_326);
            // if dot >= 0.0:                                                                 <L 206>
            var_328 = (var_325 >= var_327);
            if (var_328) {
                // v1 = v4                                                                    <L 207>
                var_329 = wp::copy(var_249);
            }
            if (!var_328) {
                // v3 = v4                                                                    <L 209>
                var_330 = wp::copy(var_249);
            }
            var_331 = wp::where(var_328, var_329, var_84);
            var_332 = wp::where(var_328, var_159, var_330);
        }
        if (!var_323) {
            // dot = wp.dot(temp1, v3.BtoA)                                                   <L 212>
            var_333 = &((var_159).BtoA);
            var_335 = wp::load(var_333);
            var_334 = wp::dot(var_316, var_335);
            // if dot >= 0.0:                                                                 <L 213>
            var_337 = (var_334 >= var_336);
            if (var_337) {
                // v2 = v4                                                                    <L 214>
                var_338 = wp::copy(var_249);
            }
            if (!var_337) {
                // v1 = v4                                                                    <L 216>
                var_339 = wp::copy(var_249);
            }
            var_340 = wp::where(var_337, var_84, var_339);
            var_341 = wp::where(var_337, var_338, var_116);
        }
        var_342 = wp::where(var_323, var_331, var_340);
        var_343 = wp::where(var_323, var_116, var_341);
        var_344 = wp::where(var_323, var_332, var_159);
        var_345 = wp::where(var_323, var_325, var_334);
        wp::assign(var_114, var_309);
        wp::assign(var_85, var_310);
        wp::assign(var_87, var_311);
        wp::assign(var_152, var_312);
        wp::assign(var_72, var_247);
        wp::assign(var_84, var_342);
        wp::assign(var_125, var_316);
        wp::assign(var_116, var_343);
        wp::assign(var_130, var_233);
        wp::assign(var_156, var_225);
        wp::assign(var_158, var_248);
        wp::assign(var_159, var_344);
        wp::assign(var_222, var_249);
    goto start_while_7;
    end_while_7:;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:77
static CUDA_CALLABLE Vert_99873389 create_solve_closest_distance__locals__simplex_get_vertex_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::int32 var_i)
{
    //---------
    // primal vars
    Vert_99873389 var_0;
    const wp::int32 var_1 = 2;
    wp::int32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    const wp::int32 var_4 = 2;
    wp::int32 var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    //---------
    // forward
    // def simplex_get_vertex(v: Mat83f, i: int) -> Vert:                                     <L 78>
    // result = Vert()                                                                        <L 86>
    var_0 = Vert_99873389();
    // result.B = v[2 * i]                                                                    <L 87>
    var_2 = wp::mul(var_1, var_i);
    var_3 = wp::extract(var_v, var_2);
    var_0.B = var_3;
    // result.BtoA = v[2 * i + 1]                                                             <L 88>
    var_5 = wp::mul(var_4, var_i);
    var_7 = wp::add(var_5, var_6);
    var_8 = wp::extract(var_v, var_7);
    var_0.BtoA = var_8;
    // return result                                                                          <L 89>
    return var_0;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:293
static CUDA_CALLABLE void create_solve_closest_distance__locals__simplex_get_closest_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::vec_t<4, wp::float32> var_barycentric,
    wp::uint32 var_usage_mask,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = 4;
    wp::range_t var_9;
    wp::int32 var_10;
    const wp::int32 var_11 = 1;
    wp::uint32 var_12;
    wp::uint32 var_13;
    wp::uint32 var_14;
    wp::uint32 var_15;
    const wp::int32 var_16 = 0;
    wp::uint32 var_17;
    bool var_18;
    Vert_99873389 var_19;
    wp::float32 var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32>* var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32>* var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    //---------
    // forward
    // def simplex_get_closest(v: Mat83f, barycentric: wp.vec4, usage_mask: wp.uint32) -> tuple[wp.vec3, wp.vec3]:       <L 294>
    // point_a = wp.vec3(0.0, 0.0, 0.0)                                                       <L 296>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // point_b = wp.vec3(0.0, 0.0, 0.0)                                                       <L 297>
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    // for i in range(4):                                                                     <L 299>
    var_9 = wp::range(var_8);
    start_for_0:;
        if (iter_cmp(var_9) == 0) goto end_for_0;
        var_10 = wp::iter_next(var_9);
        // if (usage_mask & (wp.uint32(1) << wp.uint32(i))) == wp.uint32(0):                  <L 300>
        var_12 = wp::uint32(var_11);
        var_13 = wp::uint32(var_10);
        var_14 = wp::lshift(var_12, var_13);
        var_15 = wp::bit_and(var_usage_mask, var_14);
        var_17 = wp::uint32(var_16);
        var_18 = (var_15 == var_17);
        if (var_18) {
            // continue                                                                       <L 301>
            goto start_for_0;
        }
        // vertex = simplex_get_vertex(v, i)                                                  <L 303>
        var_19 = create_solve_closest_distance__locals__simplex_get_vertex_8(var_v, var_10);
        // bc_val = barycentric[i]                                                            <L 304>
        var_20 = wp::extract(var_barycentric, var_10);
        // point_a = point_a + bc_val * (vertex.B + vertex.BtoA)                              <L 306>
        var_21 = &((var_19).B);
        var_22 = &((var_19).BtoA);
        var_24 = wp::load(var_21);
        var_25 = wp::load(var_22);
        var_23 = wp::add(var_24, var_25);
        var_26 = wp::mul(var_20, var_23);
        var_27 = wp::add(var_3, var_26);
        // point_b = point_b + bc_val * vertex.B                                              <L 307>
        var_28 = &((var_19).B);
        var_30 = wp::load(var_28);
        var_29 = wp::mul(var_20, var_30);
        var_31 = wp::add(var_7, var_29);
        wp::assign(var_3, var_27);
        wp::assign(var_7, var_31);
        goto start_for_0;
    end_for_0:;
    // return point_a, point_b                                                                <L 309>
    ret_0 = var_3;
    ret_1 = var_7;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> create_shape_support_function__locals__shape_support_25(
    GenericShapeData_ba895271 var_geom,
    wp::vec_t<3, wp::float32> var_direction,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const bool var_4 = true;
    wp::int32* var_5;
    const wp::int32 var_6 = 7;
    bool var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def shape_support(geom: Any, direction: wp.vec3, data_provider: Any) -> wp.vec3:       <L 1>
    // result = wp.vec3(0.0, 0.0, 0.0)                                                        <L 2>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // if wp.static(fuse_builtin_box_support):                                                <L 3>
    // if geom.shape_type == GeoType.BOX:                                                     <L 4>
    var_5 = &((var_geom).shape_type);
    var_8 = wp::load(var_5);
    var_7 = (var_8 == var_6);
    if (var_7) {
        // result = _support_map_box(geom, direction)                                         <L 5>
        var_9 = _support_map_box_0(var_geom, var_direction);
    }
    if (!var_7) {
        // result = support_func(geom, direction, data_provider)                              <L 7>
        var_10 = support_map_lean_0(var_geom, var_direction, var_data_provider);
    }
    var_11 = wp::where(var_7, var_9, var_10);
    // return result                                                                          <L 10>
    return var_11;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> create_support_map_function__locals__support_map_b_16(
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_direction,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def support_map_b(                                                                     <L 1>
    // tmp = wp.quat_rotate_inv(orientation_b, direction)                                     <L 22>
    var_0 = wp::quat_rotate_inv(var_orientation_b, var_direction);
    // result = shape_support(geom_b, tmp, data_provider)                                     <L 25>
    var_1 = create_shape_support_function__locals__shape_support_25(var_geom_b, var_0, var_data_provider);
    // result = wp.quat_rotate(orientation_b, result)                                         <L 28>
    var_2 = wp::quat_rotate(var_orientation_b, var_1);
    // result = result + position_b                                                           <L 29>
    var_3 = wp::add(var_2, var_position_b);
    // return result                                                                          <L 31>
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE Vert_99873389 create_support_map_function__locals__minkowski_support_16(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_direction,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::float32 var_extend,
    SupportMapDataProvider_e77f8b9f var_data_provider)
{
    //---------
    // primal vars
    Vert_99873389 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    bool var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::float32 var_8 = 0.5;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32>* var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32>* var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    //---------
    // forward
    // def minkowski_support(                                                                 <L 1>
    // v = Vert()                                                                             <L 25>
    var_0 = Vert_99873389();
    // point_a = shape_support(geom_a, direction, data_provider)                              <L 28>
    var_1 = create_shape_support_function__locals__shape_support_25(var_geom_a, var_direction, var_data_provider);
    // tmp_direction = -direction                                                             <L 31>
    var_2 = wp::neg(var_direction);
    // v.B = support_map_b(geom_b, tmp_direction, orientation_b, position_b, data_provider)       <L 32>
    var_3 = create_support_map_function__locals__support_map_b_16(var_geom_b, var_2, var_orientation_b, var_position_b, var_data_provider);
    var_0.B = var_3;
    // if extend != 0.0:                                                                      <L 35>
    var_5 = (var_extend != var_4);
    if (var_5) {
        // d = wp.normalize(direction) * extend * 0.5                                         <L 36>
        var_6 = wp::normalize(var_direction);
        var_7 = wp::mul(var_6, var_extend);
        var_9 = wp::mul(var_7, var_8);
        // point_a = point_a + d                                                              <L 37>
        var_10 = wp::add(var_1, var_9);
        // v.B = v.B - d                                                                      <L 38>
        var_11 = &((var_0).B);
        var_13 = wp::load(var_11);
        var_12 = wp::sub(var_13, var_9);
        var_0.B = var_12;
    }
    var_14 = wp::where(var_5, var_10, var_1);
    // v.BtoA = point_a - v.B                                                                 <L 41>
    var_15 = &((var_0).B);
    var_17 = wp::load(var_15);
    var_16 = wp::sub(var_14, var_17);
    var_0.BtoA = var_16;
    // return v                                                                               <L 43>
    return var_0;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:91
static CUDA_CALLABLE void create_solve_closest_distance__locals__closest_segment_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::int32 var_i0,
    wp::int32 var_i1,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<4, wp::float32> & ret_1,
    wp::uint32 & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::int32 var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 1e-08;
    bool var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 1.0;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::uint32 var_24;
    wp::uint32 var_25;
    wp::uint32 var_26;
    const wp::int32 var_27 = 1;
    wp::uint32 var_28;
    wp::uint32 var_29;
    wp::uint32 var_30;
    wp::uint32 var_31;
    const wp::float32 var_32 = 0.0;
    const wp::float32 var_33 = 0.0;
    const wp::float32 var_34 = 0.0;
    const wp::float32 var_35 = 0.0;
    wp::vec_t<4, wp::float32> var_36;
    bool var_37;
    const wp::float32 var_38 = 0.0;
    bool var_39;
    const wp::int32 var_40 = 1;
    wp::uint32 var_41;
    wp::uint32 var_42;
    wp::uint32 var_43;
    const wp::float32 var_44 = 0.0;
    const wp::float32 var_45 = 1.0;
    const wp::float32 var_46 = 0.0;
    bool var_47;
    const wp::int32 var_48 = 1;
    wp::uint32 var_49;
    wp::uint32 var_50;
    wp::uint32 var_51;
    const wp::float32 var_52 = 1.0;
    const wp::float32 var_53 = 0.0;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::uint32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::uint32 var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    //---------
    // forward
    // def closest_segment(                                                                   <L 92>
    // a = v[2 * i0 + 1]                                                                      <L 100>
    var_1 = wp::mul(var_0, var_i0);
    var_3 = wp::add(var_1, var_2);
    var_4 = wp::extract(var_v, var_3);
    // b = v[2 * i1 + 1]                                                                      <L 101>
    var_6 = wp::mul(var_5, var_i1);
    var_8 = wp::add(var_6, var_7);
    var_9 = wp::extract(var_v, var_8);
    // edge = b - a                                                                           <L 103>
    var_10 = wp::sub(var_9, var_4);
    // vsq = wp.length_sq(edge)                                                               <L 104>
    var_11 = wp::length_sq(var_10);
    // degenerate = vsq < EPSILON                                                             <L 106>
    var_13 = (var_11 < var_12);
    // denom = vsq                                                                            <L 109>
    var_14 = wp::copy(var_11);
    // if degenerate:                                                                         <L 110>
    if (var_13) {
        // denom = EPSILON                                                                    <L 111>
        var_15 = wp::copy(var_12);
    }
    var_16 = wp::where(var_13, var_15, var_14);
    // t = -wp.dot(a, edge) / denom                                                           <L 112>
    var_17 = wp::dot(var_4, var_10);
    var_18 = wp::neg(var_17);
    var_19 = wp::div(var_18, var_16);
    // lambda0 = 1.0 - t                                                                      <L 113>
    var_21 = wp::sub(var_20, var_19);
    // lambda1 = t                                                                            <L 114>
    var_22 = wp::copy(var_19);
    // mask = (wp.uint32(1) << wp.uint32(i0)) | (wp.uint32(1) << wp.uint32(i1))               <L 116>
    var_24 = wp::uint32(var_23);
    var_25 = wp::uint32(var_i0);
    var_26 = wp::lshift(var_24, var_25);
    var_28 = wp::uint32(var_27);
    var_29 = wp::uint32(var_i1);
    var_30 = wp::lshift(var_28, var_29);
    var_31 = wp::bit_or(var_26, var_30);
    // bc = wp.vec4(0.0, 0.0, 0.0, 0.0)                                                       <L 118>
    var_36 = wp::vec_t<4, wp::float32>(var_32, var_33, var_34, var_35);
    // if lambda0 < 0.0 or degenerate:                                                        <L 120>
    var_39 = (var_21 < var_38);
    var_37 = var_39;
    if (!var_37) {
        var_37 = var_37 || var_13;
    }
    if (var_37) {
        // mask = wp.uint32(1) << wp.uint32(i1)                                               <L 121>
        var_41 = wp::uint32(var_40);
        var_42 = wp::uint32(var_i1);
        var_43 = wp::lshift(var_41, var_42);
        // lambda0 = 0.0                                                                      <L 122>
        // lambda1 = 1.0                                                                      <L 123>
    }
    if (!var_37) {
        // elif lambda1 < 0.0:                                                                <L 124>
        var_47 = (var_22 < var_46);
        if (var_47) {
            // mask = wp.uint32(1) << wp.uint32(i0)                                           <L 125>
            var_49 = wp::uint32(var_48);
            var_50 = wp::uint32(var_i0);
            var_51 = wp::lshift(var_49, var_50);
            // lambda0 = 1.0                                                                  <L 126>
            // lambda1 = 0.0                                                                  <L 127>
        }
        var_54 = wp::where(var_47, var_52, var_21);
        var_55 = wp::where(var_47, var_53, var_22);
        var_56 = wp::where(var_47, var_51, var_31);
    }
    var_57 = wp::where(var_37, var_44, var_54);
    var_58 = wp::where(var_37, var_45, var_55);
    var_59 = wp::where(var_37, var_43, var_56);
    // bc[i0] = lambda0                                                                       <L 129>
    wp::assign_inplace(var_36, var_i0, var_57);
    // bc[i1] = lambda1                                                                       <L 130>
    wp::assign_inplace(var_36, var_i1, var_58);
    // return lambda0 * a + lambda1 * b, bc, mask                                             <L 132>
    var_60 = wp::mul(var_57, var_4);
    var_61 = wp::mul(var_58, var_9);
    var_62 = wp::add(var_60, var_61);
    ret_0 = var_62;
    ret_1 = var_36;
    ret_2 = var_59;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:134
static CUDA_CALLABLE void create_solve_closest_distance__locals__closest_triangle_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::int32 var_i0,
    wp::int32 var_i1,
    wp::int32 var_i2,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<4, wp::float32> & ret_1,
    wp::uint32 & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::int32 var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 2;
    wp::int32 var_11;
    const wp::int32 var_12 = 1;
    wp::int32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::float32 var_18;
    const wp::float32 var_19 = 1e-08;
    bool var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 1.0;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::float32 var_32 = 1.0;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::float32 var_35 = 1e+30;
    const wp::float32 var_36 = 0.0;
    const wp::float32 var_37 = 0.0;
    const wp::float32 var_38 = 0.0;
    wp::vec_t<3, wp::float32> var_39;
    const wp::float32 var_40 = 0.0;
    const wp::float32 var_41 = 0.0;
    const wp::float32 var_42 = 0.0;
    const wp::float32 var_43 = 0.0;
    wp::vec_t<4, wp::float32> var_44;
    const wp::int32 var_45 = 0;
    wp::uint32 var_46;
    bool var_47;
    const wp::float32 var_48 = 0.0;
    bool var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<4, wp::float32> var_51;
    wp::uint32 var_52;
    wp::float32 var_53;
    bool var_54;
    wp::vec_t<4, wp::float32> var_55;
    wp::uint32 var_56;
    wp::float32 var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::float32 var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<4, wp::float32> var_61;
    wp::uint32 var_62;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<4, wp::float32> var_65;
    wp::uint32 var_66;
    bool var_67;
    const wp::float32 var_68 = 0.0;
    bool var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<4, wp::float32> var_71;
    wp::uint32 var_72;
    wp::float32 var_73;
    bool var_74;
    wp::vec_t<4, wp::float32> var_75;
    wp::uint32 var_76;
    wp::float32 var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::float32 var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::vec_t<4, wp::float32> var_81;
    wp::uint32 var_82;
    wp::float32 var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::vec_t<4, wp::float32> var_85;
    wp::uint32 var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<4, wp::float32> var_88;
    wp::uint32 var_89;
    wp::float32 var_90;
    bool var_91;
    const wp::float32 var_92 = 0.0;
    bool var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<4, wp::float32> var_95;
    wp::uint32 var_96;
    wp::float32 var_97;
    bool var_98;
    wp::vec_t<4, wp::float32> var_99;
    wp::uint32 var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<4, wp::float32> var_103;
    wp::uint32 var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<4, wp::float32> var_106;
    wp::uint32 var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<4, wp::float32> var_109;
    wp::uint32 var_110;
    wp::float32 var_111;
    const wp::int32 var_112 = 0;
    wp::uint32 var_113;
    bool var_114;
    const wp::int32 var_115 = 1;
    wp::uint32 var_116;
    wp::uint32 var_117;
    wp::uint32 var_118;
    const wp::int32 var_119 = 1;
    wp::uint32 var_120;
    wp::uint32 var_121;
    wp::uint32 var_122;
    wp::uint32 var_123;
    const wp::int32 var_124 = 1;
    wp::uint32 var_125;
    wp::uint32 var_126;
    wp::uint32 var_127;
    wp::uint32 var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::vec_t<3, wp::float32> var_131;
    wp::vec_t<3, wp::float32> var_132;
    wp::vec_t<3, wp::float32> var_133;
    //---------
    // forward
    // def closest_triangle(                                                                  <L 135>
    // a = v[2 * i0 + 1]                                                                      <L 144>
    var_1 = wp::mul(var_0, var_i0);
    var_3 = wp::add(var_1, var_2);
    var_4 = wp::extract(var_v, var_3);
    // b = v[2 * i1 + 1]                                                                      <L 145>
    var_6 = wp::mul(var_5, var_i1);
    var_8 = wp::add(var_6, var_7);
    var_9 = wp::extract(var_v, var_8);
    // c = v[2 * i2 + 1]                                                                      <L 146>
    var_11 = wp::mul(var_10, var_i2);
    var_13 = wp::add(var_11, var_12);
    var_14 = wp::extract(var_v, var_13);
    // u = a - b                                                                              <L 148>
    var_15 = wp::sub(var_4, var_9);
    // w = a - c                                                                              <L 149>
    var_16 = wp::sub(var_4, var_14);
    // normal = wp.cross(u, w)                                                                <L 151>
    var_17 = wp::cross(var_15, var_16);
    // t = wp.length_sq(normal)                                                               <L 153>
    var_18 = wp::length_sq(var_17);
    // degenerate = t < EPSILON                                                               <L 154>
    var_20 = (var_18 < var_19);
    // denom = t                                                                              <L 156>
    var_21 = wp::copy(var_18);
    // if degenerate:                                                                         <L 157>
    if (var_20) {
        // denom = EPSILON                                                                    <L 158>
        var_22 = wp::copy(var_19);
    }
    var_23 = wp::where(var_20, var_22, var_21);
    // it = 1.0 / denom                                                                       <L 159>
    var_25 = wp::div(var_24, var_23);
    // c1 = wp.cross(u, a)                                                                    <L 161>
    var_26 = wp::cross(var_15, var_4);
    // c2 = wp.cross(a, w)                                                                    <L 162>
    var_27 = wp::cross(var_4, var_16);
    // lambda2 = wp.dot(c1, normal) * it                                                      <L 164>
    var_28 = wp::dot(var_26, var_17);
    var_29 = wp::mul(var_28, var_25);
    // lambda1 = wp.dot(c2, normal) * it                                                      <L 165>
    var_30 = wp::dot(var_27, var_17);
    var_31 = wp::mul(var_30, var_25);
    // lambda0 = 1.0 - lambda2 - lambda1                                                      <L 166>
    var_33 = wp::sub(var_32, var_29);
    var_34 = wp::sub(var_33, var_31);
    // best_distance = 1e30  # Large value                                                    <L 168>
    // closest_pt = wp.vec3(0.0, 0.0, 0.0)                                                    <L 169>
    var_39 = wp::vec_t<3, wp::float32>(var_36, var_37, var_38);
    // bc = wp.vec4(0.0, 0.0, 0.0, 0.0)                                                       <L 170>
    var_44 = wp::vec_t<4, wp::float32>(var_40, var_41, var_42, var_43);
    // mask = wp.uint32(0)                                                                    <L 171>
    var_46 = wp::uint32(var_45);
    // if lambda0 < 0.0 or degenerate:                                                        <L 174>
    var_49 = (var_34 < var_48);
    var_47 = var_49;
    if (!var_47) {
        var_47 = var_47 || var_20;
    }
    if (var_47) {
        // closest, bc_tmp, m = closest_segment(v, i1, i2)                                    <L 175>
        create_solve_closest_distance__locals__closest_segment_8(var_v, var_i1, var_i2, var_50, var_51, var_52);
        // dist = wp.length_sq(closest)                                                       <L 176>
        var_53 = wp::length_sq(var_50);
        // if dist < best_distance:                                                           <L 177>
        var_54 = (var_53 < var_35);
        if (var_54) {
            // bc = bc_tmp                                                                    <L 178>
            var_55 = wp::copy(var_51);
            // mask = m                                                                       <L 179>
            var_56 = wp::copy(var_52);
            // best_distance = dist                                                           <L 180>
            var_57 = wp::copy(var_53);
            // closest_pt = closest                                                           <L 181>
            var_58 = wp::copy(var_50);
        }
        var_59 = wp::where(var_54, var_57, var_35);
        var_60 = wp::where(var_54, var_58, var_39);
        var_61 = wp::where(var_54, var_55, var_44);
        var_62 = wp::where(var_54, var_56, var_46);
    }
    var_63 = wp::where(var_47, var_59, var_35);
    var_64 = wp::where(var_47, var_60, var_39);
    var_65 = wp::where(var_47, var_61, var_44);
    var_66 = wp::where(var_47, var_62, var_46);
    // if lambda1 < 0.0 or degenerate:                                                        <L 183>
    var_69 = (var_31 < var_68);
    var_67 = var_69;
    if (!var_67) {
        var_67 = var_67 || var_20;
    }
    if (var_67) {
        // closest, bc_tmp, m = closest_segment(v, i0, i2)                                    <L 184>
        create_solve_closest_distance__locals__closest_segment_8(var_v, var_i0, var_i2, var_70, var_71, var_72);
        // dist = wp.length_sq(closest)                                                       <L 185>
        var_73 = wp::length_sq(var_70);
        // if dist < best_distance:                                                           <L 186>
        var_74 = (var_73 < var_63);
        if (var_74) {
            // bc = bc_tmp                                                                    <L 187>
            var_75 = wp::copy(var_71);
            // mask = m                                                                       <L 188>
            var_76 = wp::copy(var_72);
            // best_distance = dist                                                           <L 189>
            var_77 = wp::copy(var_73);
            // closest_pt = closest                                                           <L 190>
            var_78 = wp::copy(var_70);
        }
        var_79 = wp::where(var_74, var_77, var_63);
        var_80 = wp::where(var_74, var_78, var_64);
        var_81 = wp::where(var_74, var_75, var_65);
        var_82 = wp::where(var_74, var_76, var_66);
    }
    var_83 = wp::where(var_67, var_79, var_63);
    var_84 = wp::where(var_67, var_80, var_64);
    var_85 = wp::where(var_67, var_81, var_65);
    var_86 = wp::where(var_67, var_82, var_66);
    var_87 = wp::where(var_67, var_70, var_50);
    var_88 = wp::where(var_67, var_71, var_51);
    var_89 = wp::where(var_67, var_72, var_52);
    var_90 = wp::where(var_67, var_73, var_53);
    // if lambda2 < 0.0 or degenerate:                                                        <L 192>
    var_93 = (var_29 < var_92);
    var_91 = var_93;
    if (!var_91) {
        var_91 = var_91 || var_20;
    }
    if (var_91) {
        // closest, bc_tmp, m = closest_segment(v, i0, i1)                                    <L 193>
        create_solve_closest_distance__locals__closest_segment_8(var_v, var_i0, var_i1, var_94, var_95, var_96);
        // dist = wp.length_sq(closest)                                                       <L 194>
        var_97 = wp::length_sq(var_94);
        // if dist < best_distance:                                                           <L 195>
        var_98 = (var_97 < var_83);
        if (var_98) {
            // bc = bc_tmp                                                                    <L 196>
            var_99 = wp::copy(var_95);
            // mask = m                                                                       <L 197>
            var_100 = wp::copy(var_96);
            // closest_pt = closest                                                           <L 198>
            var_101 = wp::copy(var_94);
        }
        var_102 = wp::where(var_98, var_101, var_84);
        var_103 = wp::where(var_98, var_99, var_85);
        var_104 = wp::where(var_98, var_100, var_86);
    }
    var_105 = wp::where(var_91, var_102, var_84);
    var_106 = wp::where(var_91, var_103, var_85);
    var_107 = wp::where(var_91, var_104, var_86);
    var_108 = wp::where(var_91, var_94, var_87);
    var_109 = wp::where(var_91, var_95, var_88);
    var_110 = wp::where(var_91, var_96, var_89);
    var_111 = wp::where(var_91, var_97, var_90);
    // if mask != wp.uint32(0):                                                               <L 200>
    var_113 = wp::uint32(var_112);
    var_114 = (var_107 != var_113);
    if (var_114) {
        // return closest_pt, bc, mask                                                        <L 201>
        ret_0 = var_105;
        ret_1 = var_106;
        ret_2 = var_107;
        return;
    }
    // bc[i0] = lambda0                                                                       <L 203>
    wp::assign_inplace(var_106, var_i0, var_34);
    // bc[i1] = lambda1                                                                       <L 204>
    wp::assign_inplace(var_106, var_i1, var_31);
    // bc[i2] = lambda2                                                                       <L 205>
    wp::assign_inplace(var_106, var_i2, var_29);
    // mask = (wp.uint32(1) << wp.uint32(i0)) | (wp.uint32(1) << wp.uint32(i1)) | (wp.uint32(1) << wp.uint32(i2))       <L 207>
    var_116 = wp::uint32(var_115);
    var_117 = wp::uint32(var_i0);
    var_118 = wp::lshift(var_116, var_117);
    var_120 = wp::uint32(var_119);
    var_121 = wp::uint32(var_i1);
    var_122 = wp::lshift(var_120, var_121);
    var_123 = wp::bit_or(var_118, var_122);
    var_125 = wp::uint32(var_124);
    var_126 = wp::uint32(var_i2);
    var_127 = wp::lshift(var_125, var_126);
    var_128 = wp::bit_or(var_123, var_127);
    // return lambda0 * a + lambda1 * b + lambda2 * c, bc, mask                               <L 208>
    var_129 = wp::mul(var_34, var_4);
    var_130 = wp::mul(var_31, var_9);
    var_131 = wp::add(var_129, var_130);
    var_132 = wp::mul(var_29, var_14);
    var_133 = wp::add(var_131, var_132);
    ret_0 = var_133;
    ret_1 = var_106;
    ret_2 = var_128;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:210
static CUDA_CALLABLE wp::float32 create_solve_closest_distance__locals__determinant_8(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_c,
    wp::vec_t<3, wp::float32> var_d)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def determinant(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> float:              <L 211>
    // return wp.dot(b - a, wp.cross(c - a, d - a))                                           <L 213>
    var_0 = wp::sub(var_b, var_a);
    var_1 = wp::sub(var_c, var_a);
    var_2 = wp::sub(var_d, var_a);
    var_3 = wp::cross(var_1, var_2);
    var_4 = wp::dot(var_0, var_3);
    return var_4;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:215
static CUDA_CALLABLE void create_solve_closest_distance__locals__closest_tetrahedron_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<4, wp::float32> & ret_1,
    wp::uint32 & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    const wp::int32 var_3 = 1;
    wp::int32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    const wp::int32 var_6 = 2;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 2;
    const wp::int32 var_13 = 2;
    wp::int32 var_14;
    const wp::int32 var_15 = 1;
    wp::int32 var_16;
    wp::vec_t<3, wp::float32> var_17;
    const wp::int32 var_18 = 2;
    const wp::int32 var_19 = 3;
    wp::int32 var_20;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::float32 var_26 = 1e-08;
    bool var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::float32 var_31 = 1.0;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.0;
    const wp::float32 var_34 = 0.0;
    const wp::float32 var_35 = 0.0;
    wp::vec_t<3, wp::float32> var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1.0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 1e+30;
    const wp::float32 var_48 = 0.0;
    const wp::float32 var_49 = 0.0;
    const wp::float32 var_50 = 0.0;
    wp::vec_t<3, wp::float32> var_51;
    const wp::float32 var_52 = 0.0;
    const wp::float32 var_53 = 0.0;
    const wp::float32 var_54 = 0.0;
    const wp::float32 var_55 = 0.0;
    wp::vec_t<4, wp::float32> var_56;
    const wp::int32 var_57 = 0;
    wp::uint32 var_58;
    bool var_59;
    const wp::float32 var_60 = 0.0;
    bool var_61;
    const wp::int32 var_62 = 1;
    const wp::int32 var_63 = 2;
    const wp::int32 var_64 = 3;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<4, wp::float32> var_66;
    wp::uint32 var_67;
    wp::float32 var_68;
    bool var_69;
    wp::vec_t<4, wp::float32> var_70;
    wp::uint32 var_71;
    wp::float32 var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::float32 var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<4, wp::float32> var_76;
    wp::uint32 var_77;
    wp::float32 var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<4, wp::float32> var_80;
    wp::uint32 var_81;
    bool var_82;
    const wp::float32 var_83 = 0.0;
    bool var_84;
    const wp::int32 var_85 = 0;
    const wp::int32 var_86 = 2;
    const wp::int32 var_87 = 3;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<4, wp::float32> var_89;
    wp::uint32 var_90;
    wp::float32 var_91;
    bool var_92;
    wp::vec_t<4, wp::float32> var_93;
    wp::uint32 var_94;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::float32 var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<4, wp::float32> var_99;
    wp::uint32 var_100;
    wp::float32 var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<4, wp::float32> var_103;
    wp::uint32 var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<4, wp::float32> var_106;
    wp::uint32 var_107;
    wp::float32 var_108;
    bool var_109;
    const wp::float32 var_110 = 0.0;
    bool var_111;
    const wp::int32 var_112 = 0;
    const wp::int32 var_113 = 1;
    const wp::int32 var_114 = 3;
    wp::vec_t<3, wp::float32> var_115;
    wp::vec_t<4, wp::float32> var_116;
    wp::uint32 var_117;
    wp::float32 var_118;
    bool var_119;
    wp::vec_t<4, wp::float32> var_120;
    wp::uint32 var_121;
    wp::float32 var_122;
    wp::vec_t<3, wp::float32> var_123;
    wp::float32 var_124;
    wp::vec_t<3, wp::float32> var_125;
    wp::vec_t<4, wp::float32> var_126;
    wp::uint32 var_127;
    wp::float32 var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::vec_t<4, wp::float32> var_130;
    wp::uint32 var_131;
    wp::vec_t<3, wp::float32> var_132;
    wp::vec_t<4, wp::float32> var_133;
    wp::uint32 var_134;
    wp::float32 var_135;
    bool var_136;
    const wp::float32 var_137 = 0.0;
    bool var_138;
    const wp::int32 var_139 = 0;
    const wp::int32 var_140 = 1;
    const wp::int32 var_141 = 2;
    wp::vec_t<3, wp::float32> var_142;
    wp::vec_t<4, wp::float32> var_143;
    wp::uint32 var_144;
    wp::float32 var_145;
    bool var_146;
    wp::vec_t<4, wp::float32> var_147;
    wp::uint32 var_148;
    wp::vec_t<3, wp::float32> var_149;
    wp::vec_t<3, wp::float32> var_150;
    wp::vec_t<4, wp::float32> var_151;
    wp::uint32 var_152;
    wp::vec_t<3, wp::float32> var_153;
    wp::vec_t<4, wp::float32> var_154;
    wp::uint32 var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::vec_t<4, wp::float32> var_157;
    wp::uint32 var_158;
    wp::float32 var_159;
    const wp::int32 var_160 = 0;
    wp::uint32 var_161;
    bool var_162;
    const wp::int32 var_163 = 0;
    const wp::int32 var_164 = 1;
    const wp::int32 var_165 = 2;
    const wp::int32 var_166 = 3;
    const wp::int32 var_167 = 15;
    wp::uint32 var_168;
    //---------
    // forward
    // def closest_tetrahedron(                                                               <L 216>
    // v0 = v[2 * 0 + 1]                                                                      <L 222>
    var_2 = wp::mul(var_0, var_1);
    var_4 = wp::add(var_2, var_3);
    var_5 = wp::extract(var_v, var_4);
    // v1 = v[2 * 1 + 1]                                                                      <L 223>
    var_8 = wp::mul(var_6, var_7);
    var_10 = wp::add(var_8, var_9);
    var_11 = wp::extract(var_v, var_10);
    // v2 = v[2 * 2 + 1]                                                                      <L 224>
    var_14 = wp::mul(var_12, var_13);
    var_16 = wp::add(var_14, var_15);
    var_17 = wp::extract(var_v, var_16);
    // v3 = v[2 * 3 + 1]                                                                      <L 225>
    var_20 = wp::mul(var_18, var_19);
    var_22 = wp::add(var_20, var_21);
    var_23 = wp::extract(var_v, var_22);
    // det_t = determinant(v0, v1, v2, v3)                                                    <L 227>
    var_24 = create_solve_closest_distance__locals__determinant_8(var_5, var_11, var_17, var_23);
    // degenerate = wp.abs(det_t) < EPSILON                                                   <L 228>
    var_25 = wp::abs(var_24);
    var_27 = (var_25 < var_26);
    // denom = det_t                                                                          <L 230>
    var_28 = wp::copy(var_24);
    // if degenerate:                                                                         <L 231>
    if (var_27) {
        // denom = EPSILON                                                                    <L 232>
        var_29 = wp::copy(var_26);
    }
    var_30 = wp::where(var_27, var_29, var_28);
    // inverse_det_t = 1.0 / denom                                                            <L 233>
    var_32 = wp::div(var_31, var_30);
    // zero = wp.vec3(0.0, 0.0, 0.0)                                                          <L 235>
    var_36 = wp::vec_t<3, wp::float32>(var_33, var_34, var_35);
    // lambda0 = determinant(zero, v1, v2, v3) * inverse_det_t                                <L 236>
    var_37 = create_solve_closest_distance__locals__determinant_8(var_36, var_11, var_17, var_23);
    var_38 = wp::mul(var_37, var_32);
    // lambda1 = determinant(v0, zero, v2, v3) * inverse_det_t                                <L 237>
    var_39 = create_solve_closest_distance__locals__determinant_8(var_5, var_36, var_17, var_23);
    var_40 = wp::mul(var_39, var_32);
    // lambda2 = determinant(v0, v1, zero, v3) * inverse_det_t                                <L 238>
    var_41 = create_solve_closest_distance__locals__determinant_8(var_5, var_11, var_36, var_23);
    var_42 = wp::mul(var_41, var_32);
    // lambda3 = 1.0 - lambda0 - lambda1 - lambda2                                            <L 239>
    var_44 = wp::sub(var_43, var_38);
    var_45 = wp::sub(var_44, var_40);
    var_46 = wp::sub(var_45, var_42);
    // best_distance = 1e30  # Large value                                                    <L 241>
    // closest_pt = wp.vec3(0.0, 0.0, 0.0)                                                    <L 242>
    var_51 = wp::vec_t<3, wp::float32>(var_48, var_49, var_50);
    // bc = wp.vec4(0.0, 0.0, 0.0, 0.0)                                                       <L 243>
    var_56 = wp::vec_t<4, wp::float32>(var_52, var_53, var_54, var_55);
    // mask = wp.uint32(0)                                                                    <L 244>
    var_58 = wp::uint32(var_57);
    // if lambda0 < 0.0 or degenerate:                                                        <L 247>
    var_61 = (var_38 < var_60);
    var_59 = var_61;
    if (!var_59) {
        var_59 = var_59 || var_27;
    }
    if (var_59) {
        // closest, bc_tmp, m = closest_triangle(v, 1, 2, 3)                                  <L 248>
        create_solve_closest_distance__locals__closest_triangle_8(var_v, var_62, var_63, var_64, var_65, var_66, var_67);
        // dist = wp.length_sq(closest)                                                       <L 249>
        var_68 = wp::length_sq(var_65);
        // if dist < best_distance:                                                           <L 250>
        var_69 = (var_68 < var_47);
        if (var_69) {
            // bc = bc_tmp                                                                    <L 251>
            var_70 = wp::copy(var_66);
            // mask = m                                                                       <L 252>
            var_71 = wp::copy(var_67);
            // best_distance = dist                                                           <L 253>
            var_72 = wp::copy(var_68);
            // closest_pt = closest                                                           <L 254>
            var_73 = wp::copy(var_65);
        }
        var_74 = wp::where(var_69, var_72, var_47);
        var_75 = wp::where(var_69, var_73, var_51);
        var_76 = wp::where(var_69, var_70, var_56);
        var_77 = wp::where(var_69, var_71, var_58);
    }
    var_78 = wp::where(var_59, var_74, var_47);
    var_79 = wp::where(var_59, var_75, var_51);
    var_80 = wp::where(var_59, var_76, var_56);
    var_81 = wp::where(var_59, var_77, var_58);
    // if lambda1 < 0.0 or degenerate:                                                        <L 256>
    var_84 = (var_40 < var_83);
    var_82 = var_84;
    if (!var_82) {
        var_82 = var_82 || var_27;
    }
    if (var_82) {
        // closest, bc_tmp, m = closest_triangle(v, 0, 2, 3)                                  <L 257>
        create_solve_closest_distance__locals__closest_triangle_8(var_v, var_85, var_86, var_87, var_88, var_89, var_90);
        // dist = wp.length_sq(closest)                                                       <L 258>
        var_91 = wp::length_sq(var_88);
        // if dist < best_distance:                                                           <L 259>
        var_92 = (var_91 < var_78);
        if (var_92) {
            // bc = bc_tmp                                                                    <L 260>
            var_93 = wp::copy(var_89);
            // mask = m                                                                       <L 261>
            var_94 = wp::copy(var_90);
            // best_distance = dist                                                           <L 262>
            var_95 = wp::copy(var_91);
            // closest_pt = closest                                                           <L 263>
            var_96 = wp::copy(var_88);
        }
        var_97 = wp::where(var_92, var_95, var_78);
        var_98 = wp::where(var_92, var_96, var_79);
        var_99 = wp::where(var_92, var_93, var_80);
        var_100 = wp::where(var_92, var_94, var_81);
    }
    var_101 = wp::where(var_82, var_97, var_78);
    var_102 = wp::where(var_82, var_98, var_79);
    var_103 = wp::where(var_82, var_99, var_80);
    var_104 = wp::where(var_82, var_100, var_81);
    var_105 = wp::where(var_82, var_88, var_65);
    var_106 = wp::where(var_82, var_89, var_66);
    var_107 = wp::where(var_82, var_90, var_67);
    var_108 = wp::where(var_82, var_91, var_68);
    // if lambda2 < 0.0 or degenerate:                                                        <L 265>
    var_111 = (var_42 < var_110);
    var_109 = var_111;
    if (!var_109) {
        var_109 = var_109 || var_27;
    }
    if (var_109) {
        // closest, bc_tmp, m = closest_triangle(v, 0, 1, 3)                                  <L 266>
        create_solve_closest_distance__locals__closest_triangle_8(var_v, var_112, var_113, var_114, var_115, var_116, var_117);
        // dist = wp.length_sq(closest)                                                       <L 267>
        var_118 = wp::length_sq(var_115);
        // if dist < best_distance:                                                           <L 268>
        var_119 = (var_118 < var_101);
        if (var_119) {
            // bc = bc_tmp                                                                    <L 269>
            var_120 = wp::copy(var_116);
            // mask = m                                                                       <L 270>
            var_121 = wp::copy(var_117);
            // best_distance = dist                                                           <L 271>
            var_122 = wp::copy(var_118);
            // closest_pt = closest                                                           <L 272>
            var_123 = wp::copy(var_115);
        }
        var_124 = wp::where(var_119, var_122, var_101);
        var_125 = wp::where(var_119, var_123, var_102);
        var_126 = wp::where(var_119, var_120, var_103);
        var_127 = wp::where(var_119, var_121, var_104);
    }
    var_128 = wp::where(var_109, var_124, var_101);
    var_129 = wp::where(var_109, var_125, var_102);
    var_130 = wp::where(var_109, var_126, var_103);
    var_131 = wp::where(var_109, var_127, var_104);
    var_132 = wp::where(var_109, var_115, var_105);
    var_133 = wp::where(var_109, var_116, var_106);
    var_134 = wp::where(var_109, var_117, var_107);
    var_135 = wp::where(var_109, var_118, var_108);
    // if lambda3 < 0.0 or degenerate:                                                        <L 274>
    var_138 = (var_46 < var_137);
    var_136 = var_138;
    if (!var_136) {
        var_136 = var_136 || var_27;
    }
    if (var_136) {
        // closest, bc_tmp, m = closest_triangle(v, 0, 1, 2)                                  <L 275>
        create_solve_closest_distance__locals__closest_triangle_8(var_v, var_139, var_140, var_141, var_142, var_143, var_144);
        // dist = wp.length_sq(closest)                                                       <L 276>
        var_145 = wp::length_sq(var_142);
        // if dist < best_distance:                                                           <L 277>
        var_146 = (var_145 < var_128);
        if (var_146) {
            // bc = bc_tmp                                                                    <L 278>
            var_147 = wp::copy(var_143);
            // mask = m                                                                       <L 279>
            var_148 = wp::copy(var_144);
            // closest_pt = closest                                                           <L 280>
            var_149 = wp::copy(var_142);
        }
        var_150 = wp::where(var_146, var_149, var_129);
        var_151 = wp::where(var_146, var_147, var_130);
        var_152 = wp::where(var_146, var_148, var_131);
    }
    var_153 = wp::where(var_136, var_150, var_129);
    var_154 = wp::where(var_136, var_151, var_130);
    var_155 = wp::where(var_136, var_152, var_131);
    var_156 = wp::where(var_136, var_142, var_132);
    var_157 = wp::where(var_136, var_143, var_133);
    var_158 = wp::where(var_136, var_144, var_134);
    var_159 = wp::where(var_136, var_145, var_135);
    // if mask != wp.uint32(0):                                                               <L 282>
    var_161 = wp::uint32(var_160);
    var_162 = (var_155 != var_161);
    if (var_162) {
        // return closest_pt, bc, mask                                                        <L 283>
        ret_0 = var_153;
        ret_1 = var_154;
        ret_2 = var_155;
        return;
    }
    // bc[0] = lambda0                                                                        <L 285>
    wp::assign_inplace(var_154, var_163, var_38);
    // bc[1] = lambda1                                                                        <L 286>
    wp::assign_inplace(var_154, var_164, var_40);
    // bc[2] = lambda2                                                                        <L 287>
    wp::assign_inplace(var_154, var_165, var_42);
    // bc[3] = lambda3                                                                        <L 288>
    wp::assign_inplace(var_154, var_166, var_46);
    // mask = wp.uint32(15)  # 0b1111                                                         <L 290>
    var_168 = wp::uint32(var_167);
    // return zero, bc, mask                                                                  <L 291>
    ret_0 = var_36;
    ret_1 = var_154;
    ret_2 = var_168;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:0
static CUDA_CALLABLE void create_solve_closest_distance__locals__solve_closest_distance_core_8(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::float32 var_extend,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    wp::int32 var_MAX_ITER,
    wp::float32 var_COLLIDE_EPSILON,
    bool & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::float32 & ret_4)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    wp::vec_t<3, wp::float32> var_5;
    const wp::float32 var_6 = 0.0;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    wp::vec_t<3, wp::float32> var_9;
    const wp::float32 var_10 = 0.0;
    const wp::float32 var_11 = 0.0;
    const wp::float32 var_12 = 0.0;
    wp::vec_t<3, wp::float32> var_13;
    wp::mat_t<8, 3, wp::float32> var_14;
    const wp::float32 var_15 = 0.0;
    const wp::float32 var_16 = 0.0;
    const wp::float32 var_17 = 0.0;
    const wp::float32 var_18 = 0.0;
    wp::vec_t<4, wp::float32> var_19;
    const wp::int32 var_20 = 0;
    wp::uint32 var_21;
    wp::int32 var_22;
    MinkowskiCenter_f7efefbe var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 1.0;
    const wp::float32 var_29 = 0.0;
    const wp::float32 var_30 = 0.0;
    wp::vec_t<3, wp::float32> var_31;
    const wp::int32 var_32 = 0;
    bool var_33;
    const wp::int32 var_34 = 1;
    wp::int32 var_35;
    wp::float32 var_36;
    bool var_37;
    const wp::float32 var_38 = 0.0;
    const wp::float32 var_39 = 0.0;
    const wp::float32 var_40 = 0.0;
    const wp::float32 var_41 = 0.0;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    const bool var_45 = false;
    wp::float32 var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    Vert_99873389 var_52;
    wp::vec_t<3, wp::float32>* var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    bool var_58;
    const wp::float32 var_59 = 0.0;
    bool var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    bool var_64;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::int32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    const bool var_71 = false;
    bool var_72;
    const wp::int32 var_73 = 4;
    wp::range_t var_74;
    wp::int32 var_75;
    const wp::int32 var_76 = 1;
    wp::uint32 var_77;
    wp::uint32 var_78;
    wp::uint32 var_79;
    wp::uint32 var_80;
    const wp::int32 var_81 = 0;
    wp::uint32 var_82;
    bool var_83;
    const wp::int32 var_84 = 2;
    wp::int32 var_85;
    const wp::int32 var_86 = 1;
    wp::int32 var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    bool var_92;
    const bool var_93 = true;
    bool var_94;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::int32 var_99;
    wp::vec_t<3, wp::float32> var_100;
    const wp::int32 var_101 = 0;
    const wp::int32 var_102 = 0;
    const wp::int32 var_103 = 0;
    wp::vec_t<4, wp::int32> var_104;
    const wp::int32 var_105 = 0;
    const wp::int32 var_106 = 1;
    wp::uint32 var_107;
    wp::uint32 var_108;
    wp::uint32 var_109;
    wp::uint32 var_110;
    const wp::int32 var_111 = 0;
    wp::uint32 var_112;
    bool var_113;
    const wp::int32 var_114 = 1;
    wp::int32 var_115;
    wp::int32 var_116;
    wp::int32 var_117;
    wp::int32 var_118;
    const wp::int32 var_119 = 1;
    const wp::int32 var_120 = 1;
    wp::uint32 var_121;
    wp::uint32 var_122;
    wp::uint32 var_123;
    wp::uint32 var_124;
    const wp::int32 var_125 = 0;
    wp::uint32 var_126;
    bool var_127;
    const wp::int32 var_128 = 1;
    wp::int32 var_129;
    wp::int32 var_130;
    wp::int32 var_131;
    wp::int32 var_132;
    const wp::int32 var_133 = 2;
    const wp::int32 var_134 = 1;
    wp::uint32 var_135;
    wp::uint32 var_136;
    wp::uint32 var_137;
    wp::uint32 var_138;
    const wp::int32 var_139 = 0;
    wp::uint32 var_140;
    bool var_141;
    const wp::int32 var_142 = 1;
    wp::int32 var_143;
    wp::int32 var_144;
    wp::int32 var_145;
    wp::int32 var_146;
    const wp::int32 var_147 = 3;
    const wp::int32 var_148 = 1;
    wp::uint32 var_149;
    wp::uint32 var_150;
    wp::uint32 var_151;
    wp::uint32 var_152;
    const wp::int32 var_153 = 0;
    wp::uint32 var_154;
    bool var_155;
    const wp::int32 var_156 = 1;
    wp::int32 var_157;
    wp::int32 var_158;
    wp::int32 var_159;
    wp::int32 var_160;
    const wp::int32 var_161 = 1;
    wp::int32 var_162;
    wp::vec_t<3, wp::float32>* var_163;
    const wp::int32 var_164 = 2;
    wp::int32 var_165;
    wp::vec_t<3, wp::float32> var_166;
    wp::vec_t<3, wp::float32>* var_167;
    const wp::int32 var_168 = 2;
    wp::int32 var_169;
    const wp::int32 var_170 = 1;
    wp::int32 var_171;
    wp::vec_t<3, wp::float32> var_172;
    const wp::float32 var_173 = 0.0;
    const wp::float32 var_174 = 0.0;
    const wp::float32 var_175 = 0.0;
    wp::vec_t<3, wp::float32> var_176;
    const bool var_177 = true;
    const wp::int32 var_178 = 1;
    bool var_179;
    const wp::int32 var_180 = 0;
    wp::int32 var_181;
    const wp::int32 var_182 = 2;
    wp::int32 var_183;
    const wp::int32 var_184 = 1;
    wp::int32 var_185;
    wp::vec_t<3, wp::float32> var_186;
    const wp::int32 var_187 = 1;
    wp::uint32 var_188;
    wp::uint32 var_189;
    wp::uint32 var_190;
    const wp::float32 var_191 = 1.0;
    const wp::int32 var_192 = 2;
    bool var_193;
    const wp::int32 var_194 = 0;
    wp::int32 var_195;
    const wp::int32 var_196 = 1;
    wp::int32 var_197;
    wp::vec_t<3, wp::float32> var_198;
    wp::vec_t<4, wp::float32> var_199;
    wp::uint32 var_200;
    wp::vec_t<4, wp::float32> var_201;
    wp::uint32 var_202;
    const wp::int32 var_203 = 3;
    bool var_204;
    const wp::int32 var_205 = 0;
    wp::int32 var_206;
    const wp::int32 var_207 = 1;
    wp::int32 var_208;
    const wp::int32 var_209 = 2;
    wp::int32 var_210;
    wp::vec_t<3, wp::float32> var_211;
    wp::vec_t<4, wp::float32> var_212;
    wp::uint32 var_213;
    wp::vec_t<4, wp::float32> var_214;
    wp::uint32 var_215;
    const wp::int32 var_216 = 4;
    bool var_217;
    wp::vec_t<3, wp::float32> var_218;
    wp::vec_t<4, wp::float32> var_219;
    wp::uint32 var_220;
    wp::vec_t<4, wp::float32> var_221;
    wp::uint32 var_222;
    const wp::int32 var_223 = 15;
    wp::uint32 var_224;
    bool var_225;
    bool var_226;
    const bool var_227 = false;
    wp::vec_t<4, wp::float32> var_228;
    wp::uint32 var_229;
    wp::vec_t<3, wp::float32> var_230;
    bool var_231;
    wp::vec_t<4, wp::float32> var_232;
    wp::uint32 var_233;
    wp::vec_t<3, wp::float32> var_234;
    bool var_235;
    wp::vec_t<4, wp::float32> var_236;
    wp::uint32 var_237;
    wp::vec_t<4, wp::float32> var_238;
    wp::uint32 var_239;
    wp::vec_t<3, wp::float32> var_240;
    bool var_241;
    wp::int32 var_242;
    wp::int32 var_243;
    wp::vec_t<4, wp::float32> var_244;
    wp::uint32 var_245;
    wp::vec_t<4, wp::float32> var_246;
    wp::uint32 var_247;
    wp::vec_t<3, wp::float32> var_248;
    bool var_249;
    wp::int32 var_250;
    wp::vec_t<3, wp::float32> var_251;
    bool var_252;
    const wp::float32 var_253 = 0.0;
    const wp::float32 var_254 = 0.0;
    const wp::float32 var_255 = 0.0;
    const wp::float32 var_256 = 0.0;
    wp::vec_t<3, wp::float32> var_257;
    wp::vec_t<3, wp::float32> var_258;
    wp::vec_t<3, wp::float32> var_259;
    const bool var_260 = false;
    wp::float32 var_261;
    wp::vec_t<3, wp::float32> var_262;
    wp::vec_t<3, wp::float32> var_263;
    wp::vec_t<3, wp::float32> var_264;
    wp::vec_t<3, wp::float32> var_265;
    wp::float32 var_266;
    wp::vec_t<3, wp::float32> var_267;
    wp::vec_t<3, wp::float32> var_268;
    wp::vec_t<3, wp::float32> var_269;
    wp::float32 var_270;
    const wp::float32 var_271 = 1e-08;
    wp::float32 var_272;
    bool var_273;
    wp::float32 var_274;
    const wp::float32 var_275 = 1.0;
    wp::float32 var_276;
    wp::vec_t<3, wp::float32> var_277;
    wp::float32 var_278;
    bool var_279;
    const wp::float32 var_280 = -1.0;
    wp::float32 var_281;
    wp::vec_t<3, wp::float32> var_282;
    wp::float32 var_283;
    const wp::float32 var_284 = 0.0;
    bool var_285;
    const wp::float32 var_286 = 1.0;
    wp::float32 var_287;
    wp::float32 var_288;
    wp::vec_t<3, wp::float32> var_289;
    const wp::float32 var_290 = 1.0;
    const wp::float32 var_291 = 0.0;
    const wp::float32 var_292 = 0.0;
    wp::vec_t<3, wp::float32> var_293;
    wp::vec_t<3, wp::float32> var_294;
    wp::vec_t<3, wp::float32> var_295;
    wp::float32 var_296;
    wp::vec_t<3, wp::float32> var_297;
    const bool var_298 = true;
    //---------
    // forward
    // def solve_closest_distance_core(                                                       <L 1>
    // distance = float(0.0)                                                                  <L 40>
    var_1 = wp::float(var_0);
    // point_a = wp.vec3(0.0, 0.0, 0.0)                                                       <L 41>
    var_5 = wp::vec_t<3, wp::float32>(var_2, var_3, var_4);
    // point_b = wp.vec3(0.0, 0.0, 0.0)                                                       <L 42>
    var_9 = wp::vec_t<3, wp::float32>(var_6, var_7, var_8);
    // normal = wp.vec3(0.0, 0.0, 0.0)                                                        <L 43>
    var_13 = wp::vec_t<3, wp::float32>(var_10, var_11, var_12);
    // simplex_v = Mat83f()                                                                   <L 46>
    var_14 = wp::mat_t<8, 3, wp::float32>();
    // simplex_barycentric = wp.vec4(0.0, 0.0, 0.0, 0.0)                                      <L 47>
    var_19 = wp::vec_t<4, wp::float32>(var_15, var_16, var_17, var_18);
    // simplex_usage_mask = wp.uint32(0)                                                      <L 48>
    var_21 = wp::uint32(var_20);
    // iter_count = int(MAX_ITER)                                                             <L 50>
    var_22 = wp::int(var_MAX_ITER);
    // center = geometric_center(geom_a, geom_b, orientation_b, position_b, data_provider)       <L 53>
    var_23 = create_shape_center_function__locals__shape_center_16(var_geom_a, var_geom_b, var_orientation_b, var_position_b, var_data_provider);
    // v = center.BtoA                                                                        <L 56>
    var_24 = &((var_23).BtoA);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // dist_sq = wp.length_sq(v)                                                              <L 57>
    var_27 = wp::length_sq(var_25);
    // last_search_dir = wp.vec3(1.0, 0.0, 0.0)                                               <L 59>
    var_31 = wp::vec_t<3, wp::float32>(var_28, var_29, var_30);
    // while iter_count > 0:                                                                  <L 61>
    start_while_0:;
    var_33 = (var_22 > var_32);
    if ((var_33) == false) goto end_while_0;
        // iter_count -= 1                                                                    <L 62>
        var_35 = wp::sub(var_22, var_34);
        // if dist_sq < COLLIDE_EPSILON * COLLIDE_EPSILON:                                    <L 64>
        var_36 = wp::mul(var_COLLIDE_EPSILON, var_COLLIDE_EPSILON);
        var_37 = (var_27 < var_36);
        if (var_37) {
            // distance = 0.0                                                                 <L 66>
            // normal = wp.vec3(0.0, 0.0, 0.0)                                                <L 67>
            var_42 = wp::vec_t<3, wp::float32>(var_39, var_40, var_41);
            // point_a, point_b = simplex_get_closest(simplex_v, simplex_barycentric, simplex_usage_mask)       <L 68>
            create_solve_closest_distance__locals__simplex_get_closest_8(var_14, var_19, var_21, var_43, var_44);
            // return False, point_a, point_b, normal, distance                               <L 69>
            ret_0 = var_45;
            ret_1 = var_43;
            ret_2 = var_44;
            ret_3 = var_42;
            ret_4 = var_38;
            return;
        }
        var_46 = wp::where(var_37, var_38, var_1);
        var_47 = wp::where(var_37, var_43, var_5);
        var_48 = wp::where(var_37, var_44, var_9);
        var_49 = wp::where(var_37, var_42, var_13);
        // search_dir = -v                                                                    <L 71>
        var_50 = wp::neg(var_25);
        // last_search_dir = search_dir                                                       <L 73>
        var_51 = wp::copy(var_50);
        // w = minkowski_support(geom_a, geom_b, search_dir, orientation_b, position_b, extend, data_provider)       <L 76>
        var_52 = create_support_map_function__locals__minkowski_support_16(var_geom_a, var_geom_b, var_50, var_orientation_b, var_position_b, var_extend, var_data_provider);
        // w_v = w.BtoA                                                                       <L 80>
        var_53 = &((var_52).BtoA);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // delta_dist = wp.dot(v, v - w_v)                                                    <L 81>
        var_56 = wp::sub(var_25, var_54);
        var_57 = wp::dot(var_25, var_56);
        // if delta_dist <= 0.0 or delta_dist * delta_dist < (COLLIDE_EPSILON * COLLIDE_EPSILON * dist_sq):       <L 82>
        var_60 = (var_57 <= var_59);
        var_58 = var_60;
        if (!var_58) {
            var_61 = wp::mul(var_57, var_57);
            var_62 = wp::mul(var_COLLIDE_EPSILON, var_COLLIDE_EPSILON);
            var_63 = wp::mul(var_62, var_27);
            var_64 = (var_61 < var_63);
            var_58 = var_58 || var_64;
        }
        if (var_58) {
            // break                                                                          <L 83>
            wp::assign(var_1, var_46);
            wp::assign(var_5, var_47);
            wp::assign(var_9, var_48);
            wp::assign(var_13, var_49);
            wp::assign(var_22, var_35);
            wp::assign(var_31, var_51);
            goto end_while_0;
        }
        var_65 = wp::where(var_58, var_1, var_46);
        var_66 = wp::where(var_58, var_5, var_47);
        var_67 = wp::where(var_58, var_9, var_48);
        var_68 = wp::where(var_58, var_13, var_49);
        var_69 = wp::where(var_58, var_22, var_35);
        var_70 = wp::where(var_58, var_31, var_51);
        // is_duplicate = bool(False)                                                         <L 86>
        var_72 = bool(var_71);
        // for i in range(4):                                                                 <L 87>
        var_74 = wp::range(var_73);
        start_for_3:;
            if (iter_cmp(var_74) == 0) goto end_for_3;
            var_75 = wp::iter_next(var_74);
            // if (simplex_usage_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0):       <L 88>
            var_77 = wp::uint32(var_76);
            var_78 = wp::uint32(var_75);
            var_79 = wp::lshift(var_77, var_78);
            var_80 = wp::bit_and(var_21, var_79);
            var_82 = wp::uint32(var_81);
            var_83 = (var_80 != var_82);
            if (var_83) {
                // if wp.length_sq(simplex_v[2 * i + 1] - w_v) < COLLIDE_EPSILON * COLLIDE_EPSILON:       <L 90>
                var_85 = wp::mul(var_84, var_75);
                var_87 = wp::add(var_85, var_86);
                var_88 = wp::extract(var_14, var_87);
                var_89 = wp::sub(var_88, var_54);
                var_90 = wp::length_sq(var_89);
                var_91 = wp::mul(var_COLLIDE_EPSILON, var_COLLIDE_EPSILON);
                var_92 = (var_90 < var_91);
                if (var_92) {
                    // is_duplicate = bool(True)                                              <L 91>
                    var_94 = bool(var_93);
                    // break                                                                  <L 92>
                    wp::assign(var_72, var_94);
                    goto end_for_3;
                }
            }
            goto start_for_3;
        end_for_3:;
        // if is_duplicate:                                                                   <L 93>
        if (var_72) {
            // break                                                                          <L 94>
            wp::assign(var_1, var_65);
            wp::assign(var_5, var_66);
            wp::assign(var_9, var_67);
            wp::assign(var_13, var_68);
            wp::assign(var_22, var_69);
            wp::assign(var_31, var_70);
            goto end_while_0;
        }
        var_95 = wp::where(var_72, var_1, var_65);
        var_96 = wp::where(var_72, var_5, var_66);
        var_97 = wp::where(var_72, var_9, var_67);
        var_98 = wp::where(var_72, var_13, var_68);
        var_99 = wp::where(var_72, var_22, var_69);
        var_100 = wp::where(var_72, var_31, var_70);
        // use_count = 0                                                                      <L 98>
        // free_slot = 0                                                                      <L 99>
        // indices = wp.vec4i(0)                                                              <L 100>
        var_104 = wp::vec_t<4, wp::int32>(var_103);
        // for i in range(4):                                                                 <L 102>
        // if (simplex_usage_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0):          <L 103>
        var_107 = wp::uint32(var_106);
        var_108 = wp::uint32(var_105);
        var_109 = wp::lshift(var_107, var_108);
        var_110 = wp::bit_and(var_21, var_109);
        var_112 = wp::uint32(var_111);
        var_113 = (var_110 != var_112);
        if (var_113) {
            // indices[use_count] = i                                                         <L 104>
            wp::assign_inplace(var_104, var_101, var_105);
            // use_count += 1                                                                 <L 105>
            var_115 = wp::add(var_101, var_114);
        }
        if (!var_113) {
            // free_slot = i                                                                  <L 107>
            var_116 = wp::copy(var_105);
        }
        var_117 = wp::where(var_113, var_115, var_101);
        var_118 = wp::where(var_113, var_102, var_116);
        // if (simplex_usage_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0):          <L 103>
        var_121 = wp::uint32(var_120);
        var_122 = wp::uint32(var_119);
        var_123 = wp::lshift(var_121, var_122);
        var_124 = wp::bit_and(var_21, var_123);
        var_126 = wp::uint32(var_125);
        var_127 = (var_124 != var_126);
        if (var_127) {
            // indices[use_count] = i                                                         <L 104>
            wp::assign_inplace(var_104, var_117, var_119);
            // use_count += 1                                                                 <L 105>
            var_129 = wp::add(var_117, var_128);
        }
        if (!var_127) {
            // free_slot = i                                                                  <L 107>
            var_130 = wp::copy(var_119);
        }
        var_131 = wp::where(var_127, var_129, var_117);
        var_132 = wp::where(var_127, var_118, var_130);
        // if (simplex_usage_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0):          <L 103>
        var_135 = wp::uint32(var_134);
        var_136 = wp::uint32(var_133);
        var_137 = wp::lshift(var_135, var_136);
        var_138 = wp::bit_and(var_21, var_137);
        var_140 = wp::uint32(var_139);
        var_141 = (var_138 != var_140);
        if (var_141) {
            // indices[use_count] = i                                                         <L 104>
            wp::assign_inplace(var_104, var_131, var_133);
            // use_count += 1                                                                 <L 105>
            var_143 = wp::add(var_131, var_142);
        }
        if (!var_141) {
            // free_slot = i                                                                  <L 107>
            var_144 = wp::copy(var_133);
        }
        var_145 = wp::where(var_141, var_143, var_131);
        var_146 = wp::where(var_141, var_132, var_144);
        // if (simplex_usage_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0):          <L 103>
        var_149 = wp::uint32(var_148);
        var_150 = wp::uint32(var_147);
        var_151 = wp::lshift(var_149, var_150);
        var_152 = wp::bit_and(var_21, var_151);
        var_154 = wp::uint32(var_153);
        var_155 = (var_152 != var_154);
        if (var_155) {
            // indices[use_count] = i                                                         <L 104>
            wp::assign_inplace(var_104, var_145, var_147);
            // use_count += 1                                                                 <L 105>
            var_157 = wp::add(var_145, var_156);
        }
        if (!var_155) {
            // free_slot = i                                                                  <L 107>
            var_158 = wp::copy(var_147);
        }
        var_159 = wp::where(var_155, var_157, var_145);
        var_160 = wp::where(var_155, var_146, var_158);
        // indices[use_count] = free_slot                                                     <L 109>
        wp::assign_inplace(var_104, var_159, var_160);
        // use_count += 1                                                                     <L 110>
        var_162 = wp::add(var_159, var_161);
        // simplex_v[2 * free_slot] = w.B                                                     <L 112>
        var_163 = &((var_52).B);
        var_165 = wp::mul(var_164, var_160);
        var_166 = wp::load(var_163);
        wp::assign_inplace(var_14, var_165, var_166);
        // simplex_v[2 * free_slot + 1] = w.BtoA                                              <L 113>
        var_167 = &((var_52).BtoA);
        var_169 = wp::mul(var_168, var_160);
        var_171 = wp::add(var_169, var_170);
        var_172 = wp::load(var_167);
        wp::assign_inplace(var_14, var_171, var_172);
        // closest = wp.vec3(0.0, 0.0, 0.0)                                                   <L 115>
        var_176 = wp::vec_t<3, wp::float32>(var_173, var_174, var_175);
        // success = True                                                                     <L 116>
        // if use_count == 1:                                                                 <L 118>
        var_179 = (var_162 == var_178);
        if (var_179) {
            // i0 = indices[0]                                                                <L 119>
            var_181 = wp::extract(var_104, var_180);
            // closest = simplex_v[2 * i0 + 1]                                                <L 121>
            var_183 = wp::mul(var_182, var_181);
            var_185 = wp::add(var_183, var_184);
            var_186 = wp::extract(var_14, var_185);
            // simplex_usage_mask = wp.uint32(1) << wp.uint32(i0)                             <L 122>
            var_188 = wp::uint32(var_187);
            var_189 = wp::uint32(var_181);
            var_190 = wp::lshift(var_188, var_189);
            // simplex_barycentric[i0] = 1.0                                                  <L 123>
            wp::assign_inplace(var_19, var_181, var_191);
        }
        if (!var_179) {
            // elif use_count == 2:                                                           <L 124>
            var_193 = (var_162 == var_192);
            if (var_193) {
                // i0 = indices[0]                                                            <L 125>
                var_195 = wp::extract(var_104, var_194);
                // i1 = indices[1]                                                            <L 126>
                var_197 = wp::extract(var_104, var_196);
                // closest, bc, mask = closest_segment(simplex_v, i0, i1)                     <L 127>
                create_solve_closest_distance__locals__closest_segment_8(var_14, var_195, var_197, var_198, var_199, var_200);
                // simplex_barycentric = bc                                                   <L 128>
                var_201 = wp::copy(var_199);
                // simplex_usage_mask = mask                                                  <L 129>
                var_202 = wp::copy(var_200);
            }
            if (!var_193) {
                // elif use_count == 3:                                                       <L 130>
                var_204 = (var_162 == var_203);
                if (var_204) {
                    // i0 = indices[0]                                                        <L 131>
                    var_206 = wp::extract(var_104, var_205);
                    // i1 = indices[1]                                                        <L 132>
                    var_208 = wp::extract(var_104, var_207);
                    // i2 = indices[2]                                                        <L 133>
                    var_210 = wp::extract(var_104, var_209);
                    // closest, bc, mask = closest_triangle(simplex_v, i0, i1, i2)            <L 134>
                    create_solve_closest_distance__locals__closest_triangle_8(var_14, var_206, var_208, var_210, var_211, var_212, var_213);
                    // simplex_barycentric = bc                                               <L 135>
                    var_214 = wp::copy(var_212);
                    // simplex_usage_mask = mask                                              <L 136>
                    var_215 = wp::copy(var_213);
                }
                if (!var_204) {
                    // elif use_count == 4:                                                   <L 137>
                    var_217 = (var_162 == var_216);
                    if (var_217) {
                        // closest, bc, mask = closest_tetrahedron(simplex_v)                 <L 138>
                        create_solve_closest_distance__locals__closest_tetrahedron_8(var_14, var_218, var_219, var_220);
                        // simplex_barycentric = bc                                           <L 139>
                        var_221 = wp::copy(var_219);
                        // simplex_usage_mask = mask                                          <L 140>
                        var_222 = wp::copy(var_220);
                        // inside_tetrahedron = mask == wp.uint32(15)                         <L 143>
                        var_224 = wp::uint32(var_223);
                        var_225 = (var_220 == var_224);
                        // success = not inside_tetrahedron                                   <L 144>
                        var_226 = wp::unot(var_225);
                    }
                    if (!var_217) {
                        // success = False                                                    <L 146>
                    }
                    var_228 = wp::where(var_217, var_221, var_19);
                    var_229 = wp::where(var_217, var_222, var_21);
                    var_230 = wp::where(var_217, var_218, var_176);
                    var_231 = wp::where(var_217, var_226, var_227);
                }
                var_232 = wp::where(var_204, var_214, var_228);
                var_233 = wp::where(var_204, var_215, var_229);
                var_234 = wp::where(var_204, var_211, var_230);
                var_235 = wp::where(var_204, var_177, var_231);
                var_236 = wp::where(var_204, var_212, var_219);
                var_237 = wp::where(var_204, var_213, var_220);
            }
            var_238 = wp::where(var_193, var_201, var_232);
            var_239 = wp::where(var_193, var_202, var_233);
            var_240 = wp::where(var_193, var_198, var_234);
            var_241 = wp::where(var_193, var_177, var_235);
            var_242 = wp::where(var_193, var_195, var_206);
            var_243 = wp::where(var_193, var_197, var_208);
            var_244 = wp::where(var_193, var_199, var_236);
            var_245 = wp::where(var_193, var_200, var_237);
        }
        var_246 = wp::where(var_179, var_19, var_238);
        var_247 = wp::where(var_179, var_190, var_239);
        var_248 = wp::where(var_179, var_186, var_240);
        var_249 = wp::where(var_179, var_177, var_241);
        var_250 = wp::where(var_179, var_181, var_242);
        // new_v = closest                                                                    <L 148>
        var_251 = wp::copy(var_248);
        // if not success:                                                                    <L 150>
        var_252 = wp::unot(var_249);
        if (var_252) {
            // distance = 0.0                                                                 <L 152>
            // normal = wp.vec3(0.0, 0.0, 0.0)                                                <L 153>
            var_257 = wp::vec_t<3, wp::float32>(var_254, var_255, var_256);
            // point_a, point_b = simplex_get_closest(simplex_v, simplex_barycentric, simplex_usage_mask)       <L 154>
            create_solve_closest_distance__locals__simplex_get_closest_8(var_14, var_246, var_247, var_258, var_259);
            // return False, point_a, point_b, normal, distance                               <L 155>
            ret_0 = var_260;
            ret_1 = var_258;
            ret_2 = var_259;
            ret_3 = var_257;
            ret_4 = var_253;
            return;
        }
        var_261 = wp::where(var_252, var_253, var_95);
        var_262 = wp::where(var_252, var_258, var_96);
        var_263 = wp::where(var_252, var_259, var_97);
        var_264 = wp::where(var_252, var_257, var_98);
        // v = new_v                                                                          <L 157>
        var_265 = wp::copy(var_251);
        // dist_sq = wp.length_sq(v)                                                          <L 158>
        var_266 = wp::length_sq(var_265);
        wp::assign(var_1, var_261);
        wp::assign(var_5, var_262);
        wp::assign(var_9, var_263);
        wp::assign(var_13, var_264);
        wp::assign(var_19, var_246);
        wp::assign(var_21, var_247);
        wp::assign(var_22, var_99);
        wp::assign(var_25, var_265);
        wp::assign(var_27, var_266);
        wp::assign(var_31, var_100);
    goto start_while_0;
    end_while_0:;
    // point_a, point_b = simplex_get_closest(simplex_v, simplex_barycentric, simplex_usage_mask)       <L 161>
    create_solve_closest_distance__locals__simplex_get_closest_8(var_14, var_19, var_21, var_267, var_268);
    // delta = point_b - point_a                                                              <L 164>
    var_269 = wp::sub(var_268, var_267);
    // delta_len_sq = wp.length_sq(delta)                                                     <L 165>
    var_270 = wp::length_sq(var_269);
    // if delta_len_sq > EPSILON * EPSILON:                                                   <L 166>
    var_272 = wp::mul(var_271, var_271);
    var_273 = (var_270 > var_272);
    if (var_273) {
        // distance = wp.sqrt(delta_len_sq)                                                   <L 169>
        var_274 = wp::sqrt(var_270);
        // normal = delta * (1.0 / distance)                                                  <L 170>
        var_276 = wp::div(var_275, var_274);
        var_277 = wp::mul(var_269, var_276);
    }
    if (!var_273) {
        // distance = wp.sqrt(dist_sq)                                                        <L 172>
        var_278 = wp::sqrt(var_27);
        // if distance > COLLIDE_EPSILON:                                                     <L 173>
        var_279 = (var_278 > var_COLLIDE_EPSILON);
        if (var_279) {
            // normal = v * (-1.0 / distance)                                                 <L 175>
            var_281 = wp::div(var_280, var_278);
            var_282 = wp::mul(var_25, var_281);
        }
        if (!var_279) {
            // nsq = wp.length_sq(last_search_dir)                                            <L 178>
            var_283 = wp::length_sq(var_31);
            // if nsq > 0.0:                                                                  <L 179>
            var_285 = (var_283 > var_284);
            if (var_285) {
                // normal = last_search_dir * (1.0 / wp.sqrt(nsq))                            <L 180>
                var_287 = wp::sqrt(var_283);
                var_288 = wp::div(var_286, var_287);
                var_289 = wp::mul(var_31, var_288);
            }
            if (!var_285) {
                // normal = wp.vec3(1.0, 0.0, 0.0)                                            <L 182>
                var_293 = wp::vec_t<3, wp::float32>(var_290, var_291, var_292);
            }
            var_294 = wp::where(var_285, var_289, var_293);
        }
        var_295 = wp::where(var_279, var_282, var_294);
    }
    var_296 = wp::where(var_273, var_274, var_278);
    var_297 = wp::where(var_273, var_277, var_295);
    // return True, point_a, point_b, normal, distance                                        <L 183>
    ret_0 = var_298;
    ret_1 = var_267;
    ret_2 = var_268;
    ret_3 = var_297;
    ret_4 = var_296;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:144
static CUDA_CALLABLE ContactData_40360d7c post_process_minkowski_only_0(
    ContactData_40360d7c var_contact_data,
    GenericShapeData_ba895271 var_shape_a,
    wp::vec_t<3, wp::float32> var_pos_a_adjusted,
    wp::quat_t<wp::float32> var_rot_a,
    GenericShapeData_ba895271 var_shape_b,
    wp::vec_t<3, wp::float32> var_pos_b_adjusted,
    wp::quat_t<wp::float32> var_rot_b)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::int32* var_3;
    wp::int32 var_4;
    wp::int32 var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32* var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32* var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    bool var_15;
    const wp::int32 var_16 = 3;
    bool var_17;
    const wp::int32 var_18 = 4;
    bool var_19;
    wp::vec_t<3, wp::float32>* var_20;
    const wp::float32 var_21 = 0.5;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::float32* var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    bool var_29;
    const wp::int32 var_30 = 3;
    bool var_31;
    const wp::int32 var_32 = 4;
    bool var_33;
    wp::vec_t<3, wp::float32>* var_34;
    const wp::float32 var_35 = 0.5;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::float32* var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    //---------
    // forward
    // def post_process_minkowski_only(                                                       <L 145>
    // type_a = shape_a.shape_type                                                            <L 155>
    var_0 = &((var_shape_a).shape_type);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // type_b = shape_b.shape_type                                                            <L 156>
    var_3 = &((var_shape_b).shape_type);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // normal = contact_data.contact_normal_a_to_b                                            <L 157>
    var_6 = &((var_contact_data).contact_normal_a_to_b);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // radius_eff_a = contact_data.radius_eff_a                                               <L 158>
    var_9 = &((var_contact_data).radius_eff_a);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // radius_eff_b = contact_data.radius_eff_b                                               <L 159>
    var_12 = &((var_contact_data).radius_eff_b);
    var_14 = wp::load(var_12);
    var_13 = wp::copy(var_14);
    // if type_a == GeoType.SPHERE or type_a == GeoType.CAPSULE:                              <L 161>
    var_17 = (var_1 == var_16);
    var_15 = var_17;
    if (!var_15) {
        var_19 = (var_1 == var_18);
        var_15 = var_15 || var_19;
    }
    if (var_15) {
        // contact_data.contact_point_center = contact_data.contact_point_center + normal * (radius_eff_a * 0.5)       <L 162>
        var_20 = &((var_contact_data).contact_point_center);
        var_22 = wp::mul(var_10, var_21);
        var_23 = wp::mul(var_7, var_22);
        var_25 = wp::load(var_20);
        var_24 = wp::add(var_25, var_23);
        var_contact_data.contact_point_center = var_24;
        // contact_data.contact_distance = contact_data.contact_distance - radius_eff_a       <L 163>
        var_26 = &((var_contact_data).contact_distance);
        var_28 = wp::load(var_26);
        var_27 = wp::sub(var_28, var_10);
        var_contact_data.contact_distance = var_27;
    }
    // if type_b == GeoType.SPHERE or type_b == GeoType.CAPSULE:                              <L 165>
    var_31 = (var_4 == var_30);
    var_29 = var_31;
    if (!var_29) {
        var_33 = (var_4 == var_32);
        var_29 = var_29 || var_33;
    }
    if (var_29) {
        // contact_data.contact_point_center = contact_data.contact_point_center - normal * (radius_eff_b * 0.5)       <L 166>
        var_34 = &((var_contact_data).contact_point_center);
        var_36 = wp::mul(var_13, var_35);
        var_37 = wp::mul(var_7, var_36);
        var_39 = wp::load(var_34);
        var_38 = wp::sub(var_39, var_37);
        var_contact_data.contact_point_center = var_38;
        // contact_data.contact_distance = contact_data.contact_distance - radius_eff_b       <L 167>
        var_40 = &((var_contact_data).contact_distance);
        var_42 = wp::load(var_40);
        var_41 = wp::sub(var_42, var_13);
        var_contact_data.contact_distance = var_41;
    }
    // return contact_data                                                                    <L 169>
    return var_contact_data;
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/math/__init__.py:235
static CUDA_CALLABLE void orthonormal_basis_0(
    wp::vec_t<3, wp::float32> var_n,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.0;
    bool var_5;
    const wp::float32 var_6 = 1.0;
    const wp::float32 var_7 = 1.0;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 1.0;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    const wp::int32 var_29 = 0;
    wp::float32 var_30;
    const wp::int32 var_31 = 2;
    const wp::int32 var_32 = 0;
    const wp::int32 var_33 = 1;
    wp::float32 var_34;
    const wp::int32 var_35 = 1;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::float32 var_39 = 1.0;
    wp::float32 var_40;
    const wp::int32 var_41 = 1;
    const wp::int32 var_42 = 1;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 2;
    const wp::float32 var_46 = 1.0;
    const wp::float32 var_47 = 1.0;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 0;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::int32 var_55 = 1;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 1.0;
    const wp::int32 var_60 = 0;
    wp::float32 var_61;
    const wp::int32 var_62 = 0;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::int32 var_67 = 0;
    const wp::int32 var_68 = 1;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    wp::float32 var_71;
    const wp::int32 var_72 = 2;
    const wp::int32 var_73 = 0;
    const wp::float32 var_74 = 1.0;
    const wp::int32 var_75 = 1;
    wp::float32 var_76;
    const wp::int32 var_77 = 1;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    const wp::int32 var_82 = 1;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 2;
    wp::float32 var_87;
    wp::float32 var_88;
    //---------
    // forward
    // def orthonormal_basis(n: wp.vec3):                                                     <L 236>
    // b1 = wp.vec3()                                                                         <L 252>
    var_0 = wp::vec_t<3, wp::float32>();
    // b2 = wp.vec3()                                                                         <L 253>
    var_1 = wp::vec_t<3, wp::float32>();
    // if n[2] < 0.0:                                                                         <L 254>
    var_3 = wp::extract(var_n, var_2);
    var_5 = (var_3 < var_4);
    if (var_5) {
        // a = 1.0 / (1.0 - n[2])                                                             <L 255>
        var_9 = wp::extract(var_n, var_8);
        var_10 = wp::sub(var_7, var_9);
        var_11 = wp::div(var_6, var_10);
        // b = n[0] * n[1] * a                                                                <L 256>
        var_13 = wp::extract(var_n, var_12);
        var_15 = wp::extract(var_n, var_14);
        var_16 = wp::mul(var_13, var_15);
        var_17 = wp::mul(var_16, var_11);
        // b1[0] = 1.0 - n[0] * n[0] * a                                                      <L 257>
        var_20 = wp::extract(var_n, var_19);
        var_22 = wp::extract(var_n, var_21);
        var_23 = wp::mul(var_20, var_22);
        var_24 = wp::mul(var_23, var_11);
        var_25 = wp::sub(var_18, var_24);
        wp::assign_inplace(var_0, var_26, var_25);
        // b1[1] = -b                                                                         <L 258>
        var_27 = wp::neg(var_17);
        wp::assign_inplace(var_0, var_28, var_27);
        // b1[2] = n[0]                                                                       <L 259>
        var_30 = wp::extract(var_n, var_29);
        wp::assign_inplace(var_0, var_31, var_30);
        // b2[0] = b                                                                          <L 261>
        wp::assign_inplace(var_1, var_32, var_17);
        // b2[1] = n[1] * n[1] * a - 1.0                                                      <L 262>
        var_34 = wp::extract(var_n, var_33);
        var_36 = wp::extract(var_n, var_35);
        var_37 = wp::mul(var_34, var_36);
        var_38 = wp::mul(var_37, var_11);
        var_40 = wp::sub(var_38, var_39);
        wp::assign_inplace(var_1, var_41, var_40);
        // b2[2] = -n[1]                                                                      <L 263>
        var_43 = wp::extract(var_n, var_42);
        var_44 = wp::neg(var_43);
        wp::assign_inplace(var_1, var_45, var_44);
    }
    if (!var_5) {
        // a = 1.0 / (1.0 + n[2])                                                             <L 265>
        var_49 = wp::extract(var_n, var_48);
        var_50 = wp::add(var_47, var_49);
        var_51 = wp::div(var_46, var_50);
        // b = -n[0] * n[1] * a                                                               <L 266>
        var_53 = wp::extract(var_n, var_52);
        var_54 = wp::neg(var_53);
        var_56 = wp::extract(var_n, var_55);
        var_57 = wp::mul(var_54, var_56);
        var_58 = wp::mul(var_57, var_51);
        // b1[0] = 1.0 - n[0] * n[0] * a                                                      <L 267>
        var_61 = wp::extract(var_n, var_60);
        var_63 = wp::extract(var_n, var_62);
        var_64 = wp::mul(var_61, var_63);
        var_65 = wp::mul(var_64, var_51);
        var_66 = wp::sub(var_59, var_65);
        wp::assign_inplace(var_0, var_67, var_66);
        // b1[1] = b                                                                          <L 268>
        wp::assign_inplace(var_0, var_68, var_58);
        // b1[2] = -n[0]                                                                      <L 269>
        var_70 = wp::extract(var_n, var_69);
        var_71 = wp::neg(var_70);
        wp::assign_inplace(var_0, var_72, var_71);
        // b2[0] = b                                                                          <L 271>
        wp::assign_inplace(var_1, var_73, var_58);
        // b2[1] = 1.0 - n[1] * n[1] * a                                                      <L 272>
        var_76 = wp::extract(var_n, var_75);
        var_78 = wp::extract(var_n, var_77);
        var_79 = wp::mul(var_76, var_78);
        var_80 = wp::mul(var_79, var_51);
        var_81 = wp::sub(var_74, var_80);
        wp::assign_inplace(var_1, var_82, var_81);
        // b2[2] = -n[1]                                                                      <L 273>
        var_84 = wp::extract(var_n, var_83);
        var_85 = wp::neg(var_84);
        wp::assign_inplace(var_1, var_86, var_85);
    }
    var_87 = wp::where(var_5, var_11, var_51);
    var_88 = wp::where(var_5, var_17, var_58);
    // return b1, b2                                                                          <L 275>
    ret_0 = var_0;
    ret_1 = var_1;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:623
static CUDA_CALLABLE void add_avoid_duplicates_vec2_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_arr,
    wp::int32 var_arr_count,
    wp::vec_t<2, wp::float32> var_vec,
    wp::float32 var_eps,
    wp::int32 & ret_0,
    bool & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::vec_t<2, wp::float32>* var_3;
    wp::vec_t<2, wp::float32> var_4;
    wp::vec_t<2, wp::float32> var_5;
    wp::float32 var_6;
    bool var_7;
    const bool var_8 = false;
    const wp::int32 var_9 = 1;
    bool var_10;
    const wp::int32 var_11 = 1;
    wp::int32 var_12;
    wp::vec_t<2, wp::float32>* var_13;
    wp::vec_t<2, wp::float32> var_14;
    wp::vec_t<2, wp::float32> var_15;
    wp::float32 var_16;
    bool var_17;
    const bool var_18 = false;
    const wp::int32 var_19 = 1;
    wp::int32 var_20;
    const bool var_21 = true;
    //---------
    // forward
    // def add_avoid_duplicates_vec2(arr: wp.array[wp.vec2], arr_count: int, vec: wp.vec2, eps: float) -> tuple[int, bool]:       <L 624>
    // if arr_count > 0:                                                                      <L 639>
    var_1 = (var_arr_count > var_0);
    if (var_1) {
        // if wp.length_sq(arr[0] - vec) < eps:                                               <L 640>
        var_3 = wp::address(var_arr, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_vec);
        var_6 = wp::length_sq(var_4);
        var_7 = (var_6 < var_eps);
        if (var_7) {
            // return arr_count, False                                                        <L 641>
            ret_0 = var_arr_count;
            ret_1 = var_8;
            return;
        }
    }
    // if arr_count > 1:                                                                      <L 643>
    var_10 = (var_arr_count > var_9);
    if (var_10) {
        // if wp.length_sq(arr[arr_count - 1] - vec) < eps:                                   <L 644>
        var_12 = wp::sub(var_arr_count, var_11);
        var_13 = wp::address(var_arr, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::sub(var_15, var_vec);
        var_16 = wp::length_sq(var_14);
        var_17 = (var_16 < var_eps);
        if (var_17) {
            // return arr_count, False                                                        <L 645>
            ret_0 = var_arr_count;
            ret_1 = var_18;
            return;
        }
    }
    // arr[arr_count] = vec                                                                   <L 647>
    wp::array_store(var_arr, var_arr_count, var_vec);
    // return arr_count + 1, True                                                             <L 648>
    var_20 = wp::add(var_arr_count, var_19);
    ret_0 = var_20;
    ret_1 = var_21;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:141
static CUDA_CALLABLE IncrementalPlaneTracker_8af6ef33 update_incremental_plane_tracker_0(
    IncrementalPlaneTracker_8af6ef33 var_tracker,
    wp::vec_t<3, wp::float32> var_current_point,
    wp::int32 var_current_point_id)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::int32 var_3 = 1;
    bool var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::float32 var_14;
    wp::float32* var_15;
    bool var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def update_incremental_plane_tracker(                                                  <L 142>
    // if current_point_id == 0:                                                              <L 150>
    var_1 = (var_current_point_id == var_0);
    if (var_1) {
        // tracker.reference_point = current_point                                            <L 151>
        var_tracker.reference_point = var_current_point;
        // tracker.largest_area_sq = 0.0                                                      <L 152>
        var_tracker.largest_area_sq = var_2;
    }
    if (!var_1) {
        // elif current_point_id == 1:                                                        <L 153>
        var_4 = (var_current_point_id == var_3);
        if (var_4) {
            // tracker.previous_point = current_point                                         <L 154>
            var_tracker.previous_point = var_current_point;
        }
        if (!var_4) {
            // edge1 = tracker.previous_point - tracker.reference_point                       <L 156>
            var_5 = &((var_tracker).previous_point);
            var_6 = &((var_tracker).reference_point);
            var_8 = wp::load(var_5);
            var_9 = wp::load(var_6);
            var_7 = wp::sub(var_8, var_9);
            // edge2 = current_point - tracker.reference_point                                <L 157>
            var_10 = &((var_tracker).reference_point);
            var_12 = wp::load(var_10);
            var_11 = wp::sub(var_current_point, var_12);
            // cross = wp.cross(edge1, edge2)                                                 <L 158>
            var_13 = wp::cross(var_7, var_11);
            // area_sq = wp.dot(cross, cross)                                                 <L 159>
            var_14 = wp::dot(var_13, var_13);
            // if area_sq > tracker.largest_area_sq:                                          <L 160>
            var_15 = &((var_tracker).largest_area_sq);
            var_17 = wp::load(var_15);
            var_16 = (var_14 > var_17);
            if (var_16) {
                // tracker.largest_area_sq = area_sq                                          <L 161>
                var_tracker.largest_area_sq = var_14;
                // tracker.normal = cross                                                     <L 162>
                var_tracker.normal = var_13;
            }
            // tracker.previous_point = current_point                                         <L 163>
            var_tracker.previous_point = var_current_point;
        }
    }
    // return tracker                                                                         <L 164>
    return var_tracker;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:167
static CUDA_CALLABLE wp::vec_t<3, wp::float32> compute_line_segment_projector_normal_0(
    wp::vec_t<3, wp::float32> var_segment_dir,
    wp::vec_t<3, wp::float32> var_reference_normal)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 1e-12;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    //---------
    // forward
    // def compute_line_segment_projector_normal(                                             <L 168>
    // right = wp.cross(segment_dir, reference_normal)                                        <L 183>
    var_0 = wp::cross(var_segment_dir, var_reference_normal);
    // normal = wp.cross(right, segment_dir)                                                  <L 184>
    var_1 = wp::cross(var_0, var_segment_dir);
    // length = wp.length(normal)                                                             <L 185>
    var_2 = wp::length(var_1);
    // return normal * (1.0 / length) if length > 1.0e-12 else reference_normal               <L 186>
    var_4 = (var_2 > var_3);
    if (var_4) {
        var_6 = wp::div(var_5, var_2);
        var_7 = wp::mul(var_1, var_6);
    }
    if (!var_4) {
    }
    var_8 = wp::where(var_4, var_7, var_reference_normal);
    return var_8;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:189
static CUDA_CALLABLE void create_body_projectors_0(
    IncrementalPlaneTracker_8af6ef33 var_plane_tracker_a,
    wp::vec_t<3, wp::float32> var_anchor_point_a,
    IncrementalPlaneTracker_8af6ef33 var_plane_tracker_b,
    wp::vec_t<3, wp::float32> var_anchor_point_b,
    wp::vec_t<3, wp::float32> var_contact_normal,
    BodyProjector_d067bb7a & ret_0,
    BodyProjector_d067bb7a & ret_1)
{
    //---------
    // primal vars
    BodyProjector_d067bb7a var_0;
    BodyProjector_d067bb7a var_1;
    bool var_2;
    wp::float32* var_3;
    const wp::float32 var_4 = 0.0;
    bool var_5;
    wp::float32 var_6;
    wp::float32* var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32>* var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::float32 var_21 = 0.5;
    wp::vec_t<3, wp::float32>* var_22;
    wp::vec_t<3, wp::float32>* var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.5;
    wp::vec_t<3, wp::float32>* var_34;
    wp::vec_t<3, wp::float32>* var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::float32 var_44;
    wp::float32* var_45;
    const wp::float32 var_46 = 0.0;
    bool var_47;
    wp::float32 var_48;
    const wp::float32 var_49 = 1.0;
    const wp::float32 var_50 = 1e-12;
    wp::float32* var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::vec_t<3, wp::float32>* var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32>* var_59;
    wp::float32 var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::float32 var_62;
    wp::float32* var_63;
    const wp::float32 var_64 = 0.0;
    bool var_65;
    wp::float32 var_66;
    const wp::float32 var_67 = 1.0;
    const wp::float32 var_68 = 1e-12;
    wp::float32* var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::vec_t<3, wp::float32>* var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32>* var_77;
    wp::float32 var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32* var_82;
    const wp::float32 var_83 = 0.0;
    bool var_84;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32>* var_86;
    wp::vec_t<3, wp::float32>* var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    const wp::float32 var_91 = 0.5;
    wp::vec_t<3, wp::float32>* var_92;
    wp::vec_t<3, wp::float32>* var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32>* var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32>* var_101;
    wp::float32 var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::float32 var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::float32* var_106;
    const wp::float32 var_107 = 0.0;
    bool var_108;
    wp::float32 var_109;
    wp::vec_t<3, wp::float32>* var_110;
    wp::vec_t<3, wp::float32>* var_111;
    wp::vec_t<3, wp::float32> var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32> var_114;
    const wp::float32 var_115 = 0.5;
    wp::vec_t<3, wp::float32>* var_116;
    wp::vec_t<3, wp::float32>* var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::vec_t<3, wp::float32> var_119;
    wp::vec_t<3, wp::float32> var_120;
    wp::vec_t<3, wp::float32> var_121;
    wp::vec_t<3, wp::float32>* var_122;
    wp::vec_t<3, wp::float32> var_123;
    wp::vec_t<3, wp::float32> var_124;
    wp::vec_t<3, wp::float32>* var_125;
    wp::float32 var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::float32 var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::vec_t<3, wp::float32> var_130;
    //---------
    // forward
    // def create_body_projectors(                                                            <L 190>
    // projector_a = BodyProjector()                                                          <L 197>
    var_0 = BodyProjector_d067bb7a();
    // projector_b = BodyProjector()                                                          <L 198>
    var_1 = BodyProjector_d067bb7a();
    // if plane_tracker_a.largest_area_sq == 0.0 and plane_tracker_b.largest_area_sq == 0.0:       <L 200>
    var_3 = &((var_plane_tracker_a).largest_area_sq);
    var_6 = wp::load(var_3);
    var_5 = (var_6 == var_4);
    var_2 = var_5;
    if (var_2) {
        var_7 = &((var_plane_tracker_b).largest_area_sq);
        var_10 = wp::load(var_7);
        var_9 = (var_10 == var_8);
        var_2 = var_2 && var_9;
    }
    if (var_2) {
        // dir_a = plane_tracker_a.previous_point - plane_tracker_a.reference_point           <L 202>
        var_11 = &((var_plane_tracker_a).previous_point);
        var_12 = &((var_plane_tracker_a).reference_point);
        var_14 = wp::load(var_11);
        var_15 = wp::load(var_12);
        var_13 = wp::sub(var_14, var_15);
        // dir_b = plane_tracker_b.previous_point - plane_tracker_b.reference_point           <L 203>
        var_16 = &((var_plane_tracker_b).previous_point);
        var_17 = &((var_plane_tracker_b).reference_point);
        var_19 = wp::load(var_16);
        var_20 = wp::load(var_17);
        var_18 = wp::sub(var_19, var_20);
        // point_on_plane_a = 0.5 * (plane_tracker_a.reference_point + plane_tracker_a.previous_point)       <L 205>
        var_22 = &((var_plane_tracker_a).reference_point);
        var_23 = &((var_plane_tracker_a).previous_point);
        var_25 = wp::load(var_22);
        var_26 = wp::load(var_23);
        var_24 = wp::add(var_25, var_26);
        var_27 = wp::mul(var_21, var_24);
        // projector_a.normal = compute_line_segment_projector_normal(dir_a, contact_normal)       <L 206>
        var_28 = compute_line_segment_projector_normal_0(var_13, var_contact_normal);
        var_0.normal = var_28;
        // projector_a.plane_d = -wp.dot(point_on_plane_a, projector_a.normal)                <L 207>
        var_29 = &((var_0).normal);
        var_31 = wp::load(var_29);
        var_30 = wp::dot(var_27, var_31);
        var_32 = wp::neg(var_30);
        var_0.plane_d = var_32;
        // point_on_plane_b = 0.5 * (plane_tracker_b.reference_point + plane_tracker_b.previous_point)       <L 209>
        var_34 = &((var_plane_tracker_b).reference_point);
        var_35 = &((var_plane_tracker_b).previous_point);
        var_37 = wp::load(var_34);
        var_38 = wp::load(var_35);
        var_36 = wp::add(var_37, var_38);
        var_39 = wp::mul(var_33, var_36);
        // projector_b.normal = compute_line_segment_projector_normal(dir_b, contact_normal)       <L 210>
        var_40 = compute_line_segment_projector_normal_0(var_18, var_contact_normal);
        var_1.normal = var_40;
        // projector_b.plane_d = -wp.dot(point_on_plane_b, projector_b.normal)                <L 211>
        var_41 = &((var_1).normal);
        var_43 = wp::load(var_41);
        var_42 = wp::dot(var_39, var_43);
        var_44 = wp::neg(var_42);
        var_1.plane_d = var_44;
        // return projector_a, projector_b                                                    <L 213>
        ret_0 = var_0;
        ret_1 = var_1;
        return;
    }
    // if plane_tracker_a.largest_area_sq > 0.0:                                              <L 215>
    var_45 = &((var_plane_tracker_a).largest_area_sq);
    var_48 = wp::load(var_45);
    var_47 = (var_48 > var_46);
    if (var_47) {
        // inv_len_n = 1.0 / wp.sqrt(wp.max(1.0e-12, plane_tracker_a.largest_area_sq))        <L 216>
        var_51 = &((var_plane_tracker_a).largest_area_sq);
        var_53 = wp::load(var_51);
        var_52 = wp::max(var_50, var_53);
        var_54 = wp::sqrt(var_52);
        var_55 = wp::div(var_49, var_54);
        // projector_a.normal = plane_tracker_a.normal * inv_len_n                            <L 217>
        var_56 = &((var_plane_tracker_a).normal);
        var_58 = wp::load(var_56);
        var_57 = wp::mul(var_58, var_55);
        var_0.normal = var_57;
        // projector_a.plane_d = -wp.dot(anchor_point_a, projector_a.normal)                  <L 218>
        var_59 = &((var_0).normal);
        var_61 = wp::load(var_59);
        var_60 = wp::dot(var_anchor_point_a, var_61);
        var_62 = wp::neg(var_60);
        var_0.plane_d = var_62;
    }
    // if plane_tracker_b.largest_area_sq > 0.0:                                              <L 219>
    var_63 = &((var_plane_tracker_b).largest_area_sq);
    var_66 = wp::load(var_63);
    var_65 = (var_66 > var_64);
    if (var_65) {
        // inv_len_n = 1.0 / wp.sqrt(wp.max(1.0e-12, plane_tracker_b.largest_area_sq))        <L 220>
        var_69 = &((var_plane_tracker_b).largest_area_sq);
        var_71 = wp::load(var_69);
        var_70 = wp::max(var_68, var_71);
        var_72 = wp::sqrt(var_70);
        var_73 = wp::div(var_67, var_72);
        // projector_b.normal = plane_tracker_b.normal * inv_len_n                            <L 221>
        var_74 = &((var_plane_tracker_b).normal);
        var_76 = wp::load(var_74);
        var_75 = wp::mul(var_76, var_73);
        var_1.normal = var_75;
        // projector_b.plane_d = -wp.dot(anchor_point_b, projector_b.normal)                  <L 222>
        var_77 = &((var_1).normal);
        var_79 = wp::load(var_77);
        var_78 = wp::dot(var_anchor_point_b, var_79);
        var_80 = wp::neg(var_78);
        var_1.plane_d = var_80;
    }
    var_81 = wp::where(var_65, var_73, var_55);
    // if plane_tracker_a.largest_area_sq == 0.0:                                             <L 224>
    var_82 = &((var_plane_tracker_a).largest_area_sq);
    var_85 = wp::load(var_82);
    var_84 = (var_85 == var_83);
    if (var_84) {
        // dir = plane_tracker_a.previous_point - plane_tracker_a.reference_point             <L 225>
        var_86 = &((var_plane_tracker_a).previous_point);
        var_87 = &((var_plane_tracker_a).reference_point);
        var_89 = wp::load(var_86);
        var_90 = wp::load(var_87);
        var_88 = wp::sub(var_89, var_90);
        // point_on_plane_a = 0.5 * (plane_tracker_a.reference_point + plane_tracker_a.previous_point)       <L 226>
        var_92 = &((var_plane_tracker_a).reference_point);
        var_93 = &((var_plane_tracker_a).previous_point);
        var_95 = wp::load(var_92);
        var_96 = wp::load(var_93);
        var_94 = wp::add(var_95, var_96);
        var_97 = wp::mul(var_91, var_94);
        // projector_a.normal = compute_line_segment_projector_normal(dir, projector_b.normal)       <L 227>
        var_98 = &((var_1).normal);
        var_100 = wp::load(var_98);
        var_99 = compute_line_segment_projector_normal_0(var_88, var_100);
        var_0.normal = var_99;
        // projector_a.plane_d = -wp.dot(point_on_plane_a, projector_a.normal)                <L 228>
        var_101 = &((var_0).normal);
        var_103 = wp::load(var_101);
        var_102 = wp::dot(var_97, var_103);
        var_104 = wp::neg(var_102);
        var_0.plane_d = var_104;
    }
    var_105 = wp::where(var_84, var_97, var_27);
    // if plane_tracker_b.largest_area_sq == 0.0:                                             <L 230>
    var_106 = &((var_plane_tracker_b).largest_area_sq);
    var_109 = wp::load(var_106);
    var_108 = (var_109 == var_107);
    if (var_108) {
        // dir = plane_tracker_b.previous_point - plane_tracker_b.reference_point             <L 231>
        var_110 = &((var_plane_tracker_b).previous_point);
        var_111 = &((var_plane_tracker_b).reference_point);
        var_113 = wp::load(var_110);
        var_114 = wp::load(var_111);
        var_112 = wp::sub(var_113, var_114);
        // point_on_plane_b = 0.5 * (plane_tracker_b.reference_point + plane_tracker_b.previous_point)       <L 232>
        var_116 = &((var_plane_tracker_b).reference_point);
        var_117 = &((var_plane_tracker_b).previous_point);
        var_119 = wp::load(var_116);
        var_120 = wp::load(var_117);
        var_118 = wp::add(var_119, var_120);
        var_121 = wp::mul(var_115, var_118);
        // projector_b.normal = compute_line_segment_projector_normal(dir, projector_a.normal)       <L 233>
        var_122 = &((var_0).normal);
        var_124 = wp::load(var_122);
        var_123 = compute_line_segment_projector_normal_0(var_112, var_124);
        var_1.normal = var_123;
        // projector_b.plane_d = -wp.dot(point_on_plane_b, projector_b.normal)                <L 234>
        var_125 = &((var_1).normal);
        var_127 = wp::load(var_125);
        var_126 = wp::dot(var_121, var_127);
        var_128 = wp::neg(var_126);
        var_1.plane_d = var_128;
    }
    var_129 = wp::where(var_108, var_121, var_39);
    var_130 = wp::where(var_108, var_112, var_88);
    // return projector_a, projector_b                                                        <L 236>
    ret_0 = var_0;
    ret_1 = var_1;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:43
static CUDA_CALLABLE bool excess_normal_deviation_0(
    wp::vec_t<3, wp::float32> var_dir_a,
    wp::vec_t<3, wp::float32> var_dir_b)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.9993908270190958;
    bool var_3;
    //---------
    // forward
    // def excess_normal_deviation(dir_a: wp.vec3, dir_b: wp.vec3) -> bool:                   <L 44>
    // dot = wp.abs(wp.dot(dir_a, dir_b))                                                     <L 58>
    var_0 = wp::dot(var_dir_a, var_dir_b);
    var_1 = wp::abs(var_0);
    // return dot < COS_TILT_ANGLE                                                            <L 59>
    var_3 = (var_1 < var_2);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:62
static CUDA_CALLABLE wp::float32 signed_area_0(
    wp::vec_t<2, wp::float32> var_a,
    wp::vec_t<2, wp::float32> var_b,
    wp::vec_t<2, wp::float32> var_query_point)
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
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 0;
    wp::float32 var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    //---------
    // forward
    // def signed_area(a: wp.vec2, b: wp.vec2, query_point: wp.vec2) -> float:                <L 63>
    // return (b[0] - a[0]) * (query_point[1] - a[1]) - (b[1] - a[1]) * (query_point[0] - a[0])       <L 84>
    var_1 = wp::extract(var_b, var_0);
    var_3 = wp::extract(var_a, var_2);
    var_4 = wp::sub(var_1, var_3);
    var_6 = wp::extract(var_query_point, var_5);
    var_8 = wp::extract(var_a, var_7);
    var_9 = wp::sub(var_6, var_8);
    var_10 = wp::mul(var_4, var_9);
    var_12 = wp::extract(var_b, var_11);
    var_14 = wp::extract(var_a, var_13);
    var_15 = wp::sub(var_12, var_14);
    var_17 = wp::extract(var_query_point, var_16);
    var_19 = wp::extract(var_a, var_18);
    var_20 = wp::sub(var_17, var_19);
    var_21 = wp::mul(var_15, var_20);
    var_22 = wp::sub(var_10, var_21);
    return var_22;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:263
static CUDA_CALLABLE wp::vec_t<2, wp::float32> intersection_point_0(
    wp::vec_t<2, wp::float32> var_trim_seg_start,
    wp::vec_t<2, wp::float32> var_trim_seg_end,
    wp::vec_t<2, wp::float32> var_a,
    wp::vec_t<2, wp::float32> var_b)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 1.0;
    wp::float32 var_7;
    wp::vec_t<2, wp::float32> var_8;
    wp::vec_t<2, wp::float32> var_9;
    wp::vec_t<2, wp::float32> var_10;
    //---------
    // forward
    // def intersection_point(trim_seg_start: wp.vec2, trim_seg_end: wp.vec2, a: wp.vec2, b: wp.vec2) -> wp.vec2:       <L 264>
    // signed_a = signed_area(trim_seg_start, trim_seg_end, a)                                <L 281>
    var_0 = signed_area_0(var_trim_seg_start, var_trim_seg_end, var_a);
    // signed_b = signed_area(trim_seg_start, trim_seg_end, b)                                <L 282>
    var_1 = signed_area_0(var_trim_seg_start, var_trim_seg_end, var_b);
    // interp_ab = wp.abs(signed_a) / wp.abs(signed_a - signed_b)                             <L 283>
    var_2 = wp::abs(var_0);
    var_3 = wp::sub(var_0, var_1);
    var_4 = wp::abs(var_3);
    var_5 = wp::div(var_2, var_4);
    // return (1.0 - interp_ab) * a + interp_ab * b                                           <L 286>
    var_7 = wp::sub(var_6, var_5);
    var_8 = wp::mul(var_7, var_a);
    var_9 = wp::mul(var_5, var_b);
    var_10 = wp::add(var_8, var_9);
    return var_10;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:289
static CUDA_CALLABLE void insert_vec2_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_arr,
    wp::int32 var_arr_count,
    wp::int32 var_index,
    wp::vec_t<2, wp::float32> var_element)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    wp::vec_t<2, wp::float32>* var_4;
    wp::vec_t<2, wp::float32> var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    //---------
    // forward
    // def insert_vec2(arr: wp.array[wp.vec2], arr_count: int, index: int, element: wp.vec2):       <L 290>
    // i = arr_count                                                                          <L 300>
    var_0 = wp::copy(var_arr_count);
    // while i > index:                                                                       <L 301>
    start_while_0:;
    var_1 = (var_0 > var_index);
    if ((var_1) == false) goto end_while_0;
        // arr[i] = arr[i - 1]                                                                <L 302>
        var_3 = wp::sub(var_0, var_2);
        var_4 = wp::address(var_arr, var_3);
        var_5 = wp::load(var_4);
        wp::array_store(var_arr, var_0, var_5);
        // i -= 1                                                                             <L 303>
        var_7 = wp::sub(var_0, var_6);
        wp::assign(var_0, var_7);
    goto start_while_0;
    end_while_0:;
    // arr[index] = element                                                                   <L 304>
    wp::array_store(var_arr, var_index, var_element);
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:307
static CUDA_CALLABLE wp::int32 trim_in_place_0(
    wp::vec_t<2, wp::float32> var_trim_seg_start,
    wp::vec_t<2, wp::float32> var_trim_seg_end,
    wp::array_t<wp::vec_t<2, wp::float32>> var_loop,
    wp::int32 var_loop_count)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<2, wp::float32> var_4;
    const wp::int32 var_5 = -1;
    wp::int32 var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    wp::vec_t<2, wp::float32> var_9;
    const wp::int32 var_10 = -1;
    wp::int32 var_11;
    const bool var_12 = false;
    bool var_13;
    const wp::int32 var_14 = 0;
    wp::vec_t<2, wp::float32>* var_15;
    wp::float32 var_16;
    wp::vec_t<2, wp::float32> var_17;
    const wp::float32 var_18 = 0.0;
    bool var_19;
    bool var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    wp::int32 var_25;
    wp::vec_t<2, wp::float32>* var_26;
    wp::float32 var_27;
    wp::vec_t<2, wp::float32> var_28;
    const wp::float32 var_29 = 0.0;
    bool var_30;
    bool var_31;
    wp::vec_t<2, wp::float32>* var_32;
    wp::vec_t<2, wp::float32>* var_33;
    wp::vec_t<2, wp::float32> var_34;
    wp::vec_t<2, wp::float32> var_35;
    wp::vec_t<2, wp::float32> var_36;
    const wp::int32 var_37 = 0;
    bool var_38;
    wp::int32 var_39;
    bool var_40;
    wp::vec_t<2, wp::float32> var_41;
    wp::int32 var_42;
    wp::vec_t<2, wp::float32> var_43;
    wp::vec_t<2, wp::float32> var_44;
    wp::int32 var_45;
    wp::vec_t<2, wp::float32> var_46;
    wp::int32 var_47;
    bool var_48;
    wp::vec_t<2, wp::float32> var_49;
    wp::int32 var_50;
    wp::vec_t<2, wp::float32> var_51;
    wp::int32 var_52;
    bool var_53;
    bool var_54;
    bool var_55;
    const wp::int32 var_56 = 0;
    bool var_57;
    const wp::int32 var_58 = 0;
    bool var_59;
    const wp::int32 var_60 = -1;
    wp::int32 var_61;
    wp::int32 var_62;
    const wp::int32 var_63 = 0;
    wp::int32 var_64;
    bool var_65;
    const wp::int32 var_66 = 1;
    wp::int32 var_67;
    wp::vec_t<2, wp::float32>* var_68;
    wp::vec_t<2, wp::float32> var_69;
    wp::int32 var_70;
    bool var_71;
    bool var_72;
    bool var_73;
    bool var_74;
    wp::vec_t<2, wp::float32> var_75;
    bool var_76;
    bool var_77;
    bool var_78;
    const wp::int32 var_79 = 1;
    wp::int32 var_80;
    const wp::int32 var_81 = 1;
    wp::int32 var_82;
    const wp::int32 var_83 = 1;
    wp::int32 var_84;
    const wp::int32 var_85 = 1;
    wp::int32 var_86;
    const wp::int32 var_87 = 1;
    wp::int32 var_88;
    const wp::int32 var_89 = 1;
    wp::int32 var_90;
    wp::int32 var_91;
    wp::int32 var_92;
    wp::int32 var_93;
    wp::int32 var_94;
    wp::int32 var_95;
    bool var_96;
    wp::int32 var_97;
    wp::int32 var_98;
    bool var_99;
    wp::int32 var_100;
    wp::int32 var_101;
    wp::int32 var_102;
    const wp::int32 var_103 = 1;
    wp::int32 var_104;
    const wp::int32 var_105 = 1;
    wp::int32 var_106;
    const wp::int32 var_107 = 0;
    wp::int32 var_108;
    wp::int32 var_109;
    wp::int32 var_110;
    wp::int32 var_111;
    //---------
    // forward
    // def trim_in_place(                                                                     <L 308>
    // if loop_count < 3:                                                                     <L 328>
    var_1 = (var_loop_count < var_0);
    if (var_1) {
        // return loop_count                                                                  <L 329>
        return var_loop_count;
    }
    // intersection_a = wp.vec2(0.0, 0.0)                                                     <L 331>
    var_4 = wp::vec_t<2, wp::float32>(var_2, var_3);
    // change_a = int(-1)                                                                     <L 332>
    var_6 = wp::int(var_5);
    // intersection_b = wp.vec2(0.0, 0.0)                                                     <L 333>
    var_9 = wp::vec_t<2, wp::float32>(var_7, var_8);
    // change_b = int(-1)                                                                     <L 334>
    var_11 = wp::int(var_10);
    // keep = bool(False)                                                                     <L 336>
    var_13 = bool(var_12);
    // prev_outside = bool(signed_area(trim_seg_start, trim_seg_end, loop[0]) <= 0.0)         <L 339>
    var_15 = wp::address(var_loop, var_14);
    var_17 = wp::load(var_15);
    var_16 = signed_area_0(var_trim_seg_start, var_trim_seg_end, var_17);
    var_19 = (var_16 <= var_18);
    var_20 = bool(var_19);
    // for i in range(loop_count):                                                            <L 341>
    var_21 = wp::range(var_loop_count);
    start_for_1:;
        if (iter_cmp(var_21) == 0) goto end_for_1;
        var_22 = wp::iter_next(var_21);
        // next_idx = (i + 1) % loop_count                                                    <L 342>
        var_24 = wp::add(var_22, var_23);
        var_25 = wp::mod(var_24, var_loop_count);
        // outside = signed_area(trim_seg_start, trim_seg_end, loop[next_idx]) <= 0.0         <L 343>
        var_26 = wp::address(var_loop, var_25);
        var_28 = wp::load(var_26);
        var_27 = signed_area_0(var_trim_seg_start, var_trim_seg_end, var_28);
        var_30 = (var_27 <= var_29);
        // if outside != prev_outside:                                                        <L 345>
        var_31 = (var_30 != var_20);
        if (var_31) {
            // intersection = intersection_point(trim_seg_start, trim_seg_end, loop[i], loop[next_idx])       <L 346>
            var_32 = wp::address(var_loop, var_22);
            var_33 = wp::address(var_loop, var_25);
            var_35 = wp::load(var_32);
            var_36 = wp::load(var_33);
            var_34 = intersection_point_0(var_trim_seg_start, var_trim_seg_end, var_35, var_36);
            // if change_a < 0:                                                               <L 347>
            var_38 = (var_6 < var_37);
            if (var_38) {
                // change_a = i                                                               <L 348>
                var_39 = wp::copy(var_22);
                // keep = not prev_outside                                                    <L 349>
                var_40 = wp::unot(var_20);
                // intersection_a = intersection                                              <L 350>
                var_41 = wp::copy(var_34);
            }
            if (!var_38) {
                // change_b = i                                                               <L 352>
                var_42 = wp::copy(var_22);
                // intersection_b = intersection                                              <L 353>
                var_43 = wp::copy(var_34);
            }
            var_44 = wp::where(var_38, var_41, var_4);
            var_45 = wp::where(var_38, var_39, var_6);
            var_46 = wp::where(var_38, var_9, var_43);
            var_47 = wp::where(var_38, var_11, var_42);
            var_48 = wp::where(var_38, var_40, var_13);
        }
        var_49 = wp::where(var_31, var_44, var_4);
        var_50 = wp::where(var_31, var_45, var_6);
        var_51 = wp::where(var_31, var_46, var_9);
        var_52 = wp::where(var_31, var_47, var_11);
        var_53 = wp::where(var_31, var_48, var_13);
        // prev_outside = outside                                                             <L 355>
        var_54 = wp::copy(var_30);
        wp::assign(var_4, var_49);
        wp::assign(var_6, var_50);
        wp::assign(var_9, var_51);
        wp::assign(var_11, var_52);
        wp::assign(var_13, var_53);
        wp::assign(var_20, var_54);
        goto start_for_1;
    end_for_1:;
    // if change_a >= 0 and change_b >= 0:                                                    <L 357>
    var_57 = (var_6 >= var_56);
    var_55 = var_57;
    if (var_55) {
        var_59 = (var_11 >= var_58);
        var_55 = var_55 && var_59;
    }
    if (var_55) {
        // loop_indexer = int(-1)                                                             <L 358>
        var_61 = wp::int(var_60);
        // new_loop_count = int(loop_count)                                                   <L 359>
        var_62 = wp::int(var_loop_count);
        // i = int(0)                                                                         <L 361>
        var_64 = wp::int(var_63);
        // while i < loop_count:                                                              <L 362>
    start_while_3:;
        var_65 = (var_64 < var_loop_count);
    if ((var_65) == false) goto end_while_3;
            // if keep:                                                                       <L 364>
            if (var_13) {
                // loop_indexer += 1                                                          <L 365>
                var_67 = wp::add(var_61, var_66);
                // loop[loop_indexer] = loop[i]                                               <L 366>
                var_68 = wp::address(var_loop, var_64);
                var_69 = wp::load(var_68);
                wp::array_store(var_loop, var_67, var_69);
            }
            var_70 = wp::where(var_13, var_67, var_61);
            // if i == change_a or i == change_b:                                             <L 369>
            var_72 = (var_64 == var_6);
            var_71 = var_72;
            if (!var_71) {
                var_73 = (var_64 == var_11);
                var_71 = var_71 || var_73;
            }
            if (var_71) {
                // pt = intersection_a if i == change_a else intersection_b                   <L 370>
                var_74 = (var_64 == var_6);
                if (var_74) {
                }
                if (!var_74) {
                }
                var_75 = wp::where(var_74, var_4, var_9);
                // if loop_indexer == i and not keep:                                         <L 373>
                var_77 = (var_70 == var_64);
                var_76 = var_77;
                if (var_76) {
                    var_78 = wp::unot(var_13);
                    var_76 = var_76 && var_78;
                }
                if (var_76) {
                    // loop_indexer += 1                                                      <L 374>
                    var_80 = wp::add(var_70, var_79);
                    // insert_vec2(loop, new_loop_count, loop_indexer, pt)                    <L 375>
                    insert_vec2_0(var_loop, var_62, var_80, var_75);
                    // new_loop_count += 1                                                    <L 377>
                    var_82 = wp::add(var_62, var_81);
                    // i += 1                                                                 <L 378>
                    var_84 = wp::add(var_64, var_83);
                    // change_b += 1                                                          <L 379>
                    var_86 = wp::add(var_11, var_85);
                    // loop_count += 1                                                        <L 380>
                    var_88 = wp::add(var_loop_count, var_87);
                }
                if (!var_76) {
                    // loop_indexer += 1                                                      <L 382>
                    var_90 = wp::add(var_70, var_89);
                    // loop[loop_indexer] = pt                                                <L 383>
                    wp::array_store(var_loop, var_90, var_75);
                }
                var_91 = wp::where(var_76, var_88, var_loop_count);
                var_92 = wp::where(var_76, var_86, var_11);
                var_93 = wp::where(var_76, var_84, var_64);
                var_94 = wp::where(var_76, var_80, var_90);
                var_95 = wp::where(var_76, var_82, var_62);
                // keep = not keep                                                            <L 385>
                var_96 = wp::unot(var_13);
            }
            var_97 = wp::where(var_71, var_91, var_loop_count);
            var_98 = wp::where(var_71, var_92, var_11);
            var_99 = wp::where(var_71, var_96, var_13);
            var_100 = wp::where(var_71, var_93, var_64);
            var_101 = wp::where(var_71, var_94, var_70);
            var_102 = wp::where(var_71, var_95, var_62);
            // i += 1                                                                         <L 387>
            var_104 = wp::add(var_100, var_103);
            wp::assign(var_loop_count, var_97);
            wp::assign(var_11, var_98);
            wp::assign(var_13, var_99);
            wp::assign(var_64, var_104);
            wp::assign(var_61, var_101);
            wp::assign(var_62, var_102);
    goto start_while_3;
    end_while_3:;
        // new_loop_count = loop_indexer + 1                                                  <L 389>
        var_106 = wp::add(var_61, var_105);
    }
    if (!var_55) {
        // elif prev_outside:                                                                 <L 390>
        if (var_20) {
            // new_loop_count = 0                                                             <L 391>
        }
        if (!var_20) {
            // new_loop_count = loop_count                                                    <L 393>
            var_108 = wp::copy(var_loop_count);
        }
        var_109 = wp::where(var_20, var_107, var_108);
    }
    var_110 = wp::where(var_55, var_64, var_22);
    var_111 = wp::where(var_55, var_106, var_109);
    // return new_loop_count                                                                  <L 395>
    return var_111;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:398
static CUDA_CALLABLE wp::int32 trim_all_in_place_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_trim_poly,
    wp::int32 var_trim_poly_count,
    wp::array_t<wp::vec_t<2, wp::float32>> var_loop,
    wp::int32 var_loop_count)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    const wp::float32 var_4 = 1e-05;
    wp::float32 var_5;
    const wp::int32 var_6 = 2;
    bool var_7;
    const wp::int32 var_8 = 0;
    wp::vec_t<2, wp::float32>* var_9;
    wp::vec_t<2, wp::float32> var_10;
    wp::vec_t<2, wp::float32> var_11;
    const wp::int32 var_12 = 1;
    wp::vec_t<2, wp::float32>* var_13;
    wp::vec_t<2, wp::float32> var_14;
    wp::vec_t<2, wp::float32> var_15;
    const wp::int32 var_16 = 0;
    wp::float32 var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 1;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 1e-10;
    bool var_31;
    const wp::float32 var_32 = 1.0;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::int32 var_39 = 0;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::int32 var_42 = 1;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::vec_t<2, wp::float32> var_45;
    const wp::int32 var_46 = 0;
    const wp::int32 var_47 = 0;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::int32 var_50 = 1;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::vec_t<2, wp::float32> var_53;
    const wp::int32 var_54 = 1;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::vec_t<2, wp::float32> var_61;
    const wp::int32 var_62 = 2;
    const wp::int32 var_63 = 0;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::int32 var_66 = 1;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::vec_t<2, wp::float32> var_69;
    const wp::int32 var_70 = 3;
    const wp::int32 var_71 = 4;
    const wp::int32 var_72 = 1;
    wp::int32 var_73;
    wp::int32 var_74;
    wp::int32 var_75;
    const wp::int32 var_76 = 2;
    bool var_77;
    const wp::int32 var_78 = 0;
    wp::vec_t<2, wp::float32>* var_79;
    wp::vec_t<2, wp::float32> var_80;
    wp::vec_t<2, wp::float32> var_81;
    const wp::int32 var_82 = 1;
    wp::vec_t<2, wp::float32>* var_83;
    wp::vec_t<2, wp::float32> var_84;
    wp::vec_t<2, wp::float32> var_85;
    const wp::int32 var_86 = 0;
    wp::float32 var_87;
    const wp::int32 var_88 = 0;
    wp::float32 var_89;
    wp::float32 var_90;
    const wp::int32 var_91 = 1;
    wp::float32 var_92;
    const wp::int32 var_93 = 1;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    const wp::float32 var_100 = 1e-10;
    bool var_101;
    const wp::float32 var_102 = 1.0;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::int32 var_109 = 0;
    wp::float32 var_110;
    wp::float32 var_111;
    const wp::int32 var_112 = 1;
    wp::float32 var_113;
    wp::float32 var_114;
    wp::vec_t<2, wp::float32> var_115;
    const wp::int32 var_116 = 0;
    const wp::int32 var_117 = 0;
    wp::float32 var_118;
    wp::float32 var_119;
    const wp::int32 var_120 = 1;
    wp::float32 var_121;
    wp::float32 var_122;
    wp::vec_t<2, wp::float32> var_123;
    const wp::int32 var_124 = 1;
    const wp::int32 var_125 = 0;
    wp::float32 var_126;
    wp::float32 var_127;
    const wp::int32 var_128 = 1;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::vec_t<2, wp::float32> var_131;
    const wp::int32 var_132 = 2;
    const wp::int32 var_133 = 0;
    wp::float32 var_134;
    wp::float32 var_135;
    const wp::int32 var_136 = 1;
    wp::float32 var_137;
    wp::float32 var_138;
    wp::vec_t<2, wp::float32> var_139;
    const wp::int32 var_140 = 3;
    const wp::int32 var_141 = 4;
    const wp::int32 var_142 = 1;
    wp::int32 var_143;
    wp::int32 var_144;
    wp::float32 var_145;
    wp::float32 var_146;
    wp::float32 var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    wp::int32 var_150;
    wp::vec_t<2, wp::float32> var_151;
    wp::vec_t<2, wp::float32> var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::int32 var_161;
    const wp::int32 var_162 = 0;
    wp::vec_t<2, wp::float32>* var_163;
    wp::vec_t<2, wp::float32> var_164;
    wp::vec_t<2, wp::float32> var_165;
    wp::range_t var_166;
    wp::int32 var_167;
    wp::vec_t<2, wp::float32>* var_168;
    wp::vec_t<2, wp::float32> var_169;
    wp::vec_t<2, wp::float32> var_170;
    const wp::int32 var_171 = 1;
    wp::int32 var_172;
    bool var_173;
    const wp::int32 var_174 = 1;
    wp::int32 var_175;
    wp::vec_t<2, wp::float32>* var_176;
    wp::vec_t<2, wp::float32> var_177;
    wp::vec_t<2, wp::float32> var_178;
    wp::vec_t<2, wp::float32> var_179;
    wp::int32 var_180;
    //---------
    // forward
    // def trim_all_in_place(                                                                 <L 399>
    // if trim_poly_count <= 1:                                                               <L 420>
    var_1 = (var_trim_poly_count <= var_0);
    if (var_1) {
        // return wp.min(1, loop_count)  # There is no trim polygon                           <L 421>
        var_3 = wp::min(var_2, var_loop_count);
        return var_3;
    }
    // move_distance = float(1e-5)                                                            <L 423>
    var_5 = wp::float(var_4);
    // if trim_poly_count == 2:                                                               <L 425>
    var_7 = (var_trim_poly_count == var_6);
    if (var_7) {
        // p0 = trim_poly[0]                                                                  <L 427>
        var_9 = wp::address(var_trim_poly, var_8);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // p1 = trim_poly[1]                                                                  <L 428>
        var_13 = wp::address(var_trim_poly, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // dir_x = p1[0] - p0[0]                                                              <L 430>
        var_17 = wp::extract(var_14, var_16);
        var_19 = wp::extract(var_10, var_18);
        var_20 = wp::sub(var_17, var_19);
        // dir_y = p1[1] - p0[1]                                                              <L 431>
        var_22 = wp::extract(var_14, var_21);
        var_24 = wp::extract(var_10, var_23);
        var_25 = wp::sub(var_22, var_24);
        // dir_len = wp.sqrt(dir_x * dir_x + dir_y * dir_y)                                   <L 432>
        var_26 = wp::mul(var_20, var_20);
        var_27 = wp::mul(var_25, var_25);
        var_28 = wp::add(var_26, var_27);
        var_29 = wp::sqrt(var_28);
        // if dir_len > 1e-10:                                                                <L 434>
        var_31 = (var_29 > var_30);
        if (var_31) {
            // inv_dir_len = 1.0 / dir_len                                                    <L 435>
            var_33 = wp::div(var_32, var_29);
            // perp_x = -dir_y * inv_dir_len                                                  <L 436>
            var_34 = wp::neg(var_25);
            var_35 = wp::mul(var_34, var_33);
            // perp_y = dir_x * inv_dir_len                                                   <L 437>
            var_36 = wp::mul(var_20, var_33);
            // offset_x = perp_x * move_distance                                              <L 439>
            var_37 = wp::mul(var_35, var_5);
            // offset_y = perp_y * move_distance                                              <L 440>
            var_38 = wp::mul(var_36, var_5);
            // trim_poly[0] = wp.vec2(p0[0] - offset_x, p0[1] - offset_y)                     <L 442>
            var_40 = wp::extract(var_10, var_39);
            var_41 = wp::sub(var_40, var_37);
            var_43 = wp::extract(var_10, var_42);
            var_44 = wp::sub(var_43, var_38);
            var_45 = wp::vec_t<2, wp::float32>(var_41, var_44);
            wp::array_store(var_trim_poly, var_46, var_45);
            // trim_poly[1] = wp.vec2(p1[0] - offset_x, p1[1] - offset_y)                     <L 443>
            var_48 = wp::extract(var_14, var_47);
            var_49 = wp::sub(var_48, var_37);
            var_51 = wp::extract(var_14, var_50);
            var_52 = wp::sub(var_51, var_38);
            var_53 = wp::vec_t<2, wp::float32>(var_49, var_52);
            wp::array_store(var_trim_poly, var_54, var_53);
            // trim_poly[2] = wp.vec2(p1[0] + offset_x, p1[1] + offset_y)                     <L 444>
            var_56 = wp::extract(var_14, var_55);
            var_57 = wp::add(var_56, var_37);
            var_59 = wp::extract(var_14, var_58);
            var_60 = wp::add(var_59, var_38);
            var_61 = wp::vec_t<2, wp::float32>(var_57, var_60);
            wp::array_store(var_trim_poly, var_62, var_61);
            // trim_poly[3] = wp.vec2(p0[0] + offset_x, p0[1] + offset_y)                     <L 445>
            var_64 = wp::extract(var_10, var_63);
            var_65 = wp::add(var_64, var_37);
            var_67 = wp::extract(var_10, var_66);
            var_68 = wp::add(var_67, var_38);
            var_69 = wp::vec_t<2, wp::float32>(var_65, var_68);
            wp::array_store(var_trim_poly, var_70, var_69);
            // trim_poly_count = 4                                                            <L 446>
        }
        if (!var_31) {
            // return wp.min(1, loop_count)                                                   <L 448>
            var_73 = wp::min(var_72, var_loop_count);
            return var_73;
        }
        var_74 = wp::where(var_31, var_71, var_trim_poly_count);
    }
    var_75 = wp::where(var_7, var_74, var_trim_poly_count);
    // if loop_count == 2:                                                                    <L 450>
    var_77 = (var_loop_count == var_76);
    if (var_77) {
        // p0 = loop[0]                                                                       <L 452>
        var_79 = wp::address(var_loop, var_78);
        var_81 = wp::load(var_79);
        var_80 = wp::copy(var_81);
        // p1 = loop[1]                                                                       <L 453>
        var_83 = wp::address(var_loop, var_82);
        var_85 = wp::load(var_83);
        var_84 = wp::copy(var_85);
        // dir_x = p1[0] - p0[0]                                                              <L 455>
        var_87 = wp::extract(var_84, var_86);
        var_89 = wp::extract(var_80, var_88);
        var_90 = wp::sub(var_87, var_89);
        // dir_y = p1[1] - p0[1]                                                              <L 456>
        var_92 = wp::extract(var_84, var_91);
        var_94 = wp::extract(var_80, var_93);
        var_95 = wp::sub(var_92, var_94);
        // dir_len = wp.sqrt(dir_x * dir_x + dir_y * dir_y)                                   <L 457>
        var_96 = wp::mul(var_90, var_90);
        var_97 = wp::mul(var_95, var_95);
        var_98 = wp::add(var_96, var_97);
        var_99 = wp::sqrt(var_98);
        // if dir_len > 1e-10:                                                                <L 459>
        var_101 = (var_99 > var_100);
        if (var_101) {
            // inv_dir_len = 1.0 / dir_len                                                    <L 460>
            var_103 = wp::div(var_102, var_99);
            // perp_x = -dir_y * inv_dir_len                                                  <L 461>
            var_104 = wp::neg(var_95);
            var_105 = wp::mul(var_104, var_103);
            // perp_y = dir_x * inv_dir_len                                                   <L 462>
            var_106 = wp::mul(var_90, var_103);
            // offset_x = perp_x * move_distance                                              <L 464>
            var_107 = wp::mul(var_105, var_5);
            // offset_y = perp_y * move_distance                                              <L 465>
            var_108 = wp::mul(var_106, var_5);
            // loop[0] = wp.vec2(p0[0] - offset_x, p0[1] - offset_y)                          <L 467>
            var_110 = wp::extract(var_80, var_109);
            var_111 = wp::sub(var_110, var_107);
            var_113 = wp::extract(var_80, var_112);
            var_114 = wp::sub(var_113, var_108);
            var_115 = wp::vec_t<2, wp::float32>(var_111, var_114);
            wp::array_store(var_loop, var_116, var_115);
            // loop[1] = wp.vec2(p1[0] - offset_x, p1[1] - offset_y)                          <L 468>
            var_118 = wp::extract(var_84, var_117);
            var_119 = wp::sub(var_118, var_107);
            var_121 = wp::extract(var_84, var_120);
            var_122 = wp::sub(var_121, var_108);
            var_123 = wp::vec_t<2, wp::float32>(var_119, var_122);
            wp::array_store(var_loop, var_124, var_123);
            // loop[2] = wp.vec2(p1[0] + offset_x, p1[1] + offset_y)                          <L 469>
            var_126 = wp::extract(var_84, var_125);
            var_127 = wp::add(var_126, var_107);
            var_129 = wp::extract(var_84, var_128);
            var_130 = wp::add(var_129, var_108);
            var_131 = wp::vec_t<2, wp::float32>(var_127, var_130);
            wp::array_store(var_loop, var_132, var_131);
            // loop[3] = wp.vec2(p0[0] + offset_x, p0[1] + offset_y)                          <L 470>
            var_134 = wp::extract(var_80, var_133);
            var_135 = wp::add(var_134, var_107);
            var_137 = wp::extract(var_80, var_136);
            var_138 = wp::add(var_137, var_108);
            var_139 = wp::vec_t<2, wp::float32>(var_135, var_138);
            wp::array_store(var_loop, var_140, var_139);
            // loop_count = 4                                                                 <L 472>
        }
        if (!var_101) {
            // return wp.min(1, loop_count)                                                   <L 474>
            var_143 = wp::min(var_142, var_loop_count);
            return var_143;
        }
        var_144 = wp::where(var_101, var_141, var_loop_count);
        var_145 = wp::where(var_101, var_103, var_33);
        var_146 = wp::where(var_101, var_105, var_35);
        var_147 = wp::where(var_101, var_106, var_36);
        var_148 = wp::where(var_101, var_107, var_37);
        var_149 = wp::where(var_101, var_108, var_38);
    }
    var_150 = wp::where(var_77, var_144, var_loop_count);
    var_151 = wp::where(var_77, var_80, var_10);
    var_152 = wp::where(var_77, var_84, var_14);
    var_153 = wp::where(var_77, var_90, var_20);
    var_154 = wp::where(var_77, var_95, var_25);
    var_155 = wp::where(var_77, var_99, var_29);
    var_156 = wp::where(var_77, var_145, var_33);
    var_157 = wp::where(var_77, var_146, var_35);
    var_158 = wp::where(var_77, var_147, var_36);
    var_159 = wp::where(var_77, var_148, var_37);
    var_160 = wp::where(var_77, var_149, var_38);
    // current_loop_count = loop_count                                                        <L 476>
    var_161 = wp::copy(var_150);
    // trim_poly_0 = trim_poly[0]  # This allows to do more memory aliasing                   <L 478>
    var_163 = wp::address(var_trim_poly, var_162);
    var_165 = wp::load(var_163);
    var_164 = wp::copy(var_165);
    // for i in range(trim_poly_count):                                                       <L 479>
    var_166 = wp::range(var_75);
    start_for_3:;
        if (iter_cmp(var_166) == 0) goto end_for_3;
        var_167 = wp::iter_next(var_166);
        // trim_seg_start = trim_poly[i]                                                      <L 480>
        var_168 = wp::address(var_trim_poly, var_167);
        var_170 = wp::load(var_168);
        var_169 = wp::copy(var_170);
        // trim_seg_end = trim_poly_0 if i == trim_poly_count - 1 else trim_poly[i + 1]       <L 481>
        var_172 = wp::sub(var_75, var_171);
        var_173 = (var_167 == var_172);
        if (var_173) {
        }
        if (!var_173) {
            var_175 = wp::add(var_167, var_174);
            var_176 = wp::address(var_trim_poly, var_175);
            var_178 = wp::load(var_176);
            var_177 = wp::copy(var_178);
        }
        var_179 = wp::where(var_173, var_164, var_177);
        // current_loop_count = trim_in_place(trim_seg_start, trim_seg_end, loop, current_loop_count)       <L 482>
        var_180 = trim_in_place_0(var_169, var_179, var_loop, var_161);
        wp::assign(var_161, var_180);
        goto start_for_3;
    end_for_3:;
    // return current_loop_count                                                              <L 484>
    return var_161;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:583
static CUDA_CALLABLE wp::int32 remove_zero_length_edges_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_loop,
    wp::int32 var_loop_count,
    wp::float32 var_eps)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    bool var_1;
    const wp::int32 var_2 = 0;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    const wp::int32 var_5 = 1;
    wp::range_t var_6;
    wp::int32 var_7;
    wp::vec_t<2, wp::float32>* var_8;
    wp::vec_t<2, wp::float32>* var_9;
    wp::vec_t<2, wp::float32> var_10;
    wp::vec_t<2, wp::float32> var_11;
    wp::vec_t<2, wp::float32> var_12;
    wp::float32 var_13;
    bool var_14;
    const wp::int32 var_15 = 1;
    wp::int32 var_16;
    wp::vec_t<2, wp::float32>* var_17;
    wp::vec_t<2, wp::float32> var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 0;
    bool var_21;
    wp::vec_t<2, wp::float32>* var_22;
    const wp::int32 var_23 = 0;
    wp::vec_t<2, wp::float32>* var_24;
    wp::vec_t<2, wp::float32> var_25;
    wp::vec_t<2, wp::float32> var_26;
    wp::vec_t<2, wp::float32> var_27;
    wp::float32 var_28;
    bool var_29;
    wp::int32 var_30;
    const wp::int32 var_31 = 1;
    wp::int32 var_32;
    wp::int32 var_33;
    const wp::int32 var_34 = 1;
    wp::int32 var_35;
    wp::vec_t<2, wp::float32> var_36;
    wp::int32 var_37;
    const wp::int32 var_38 = 2;
    bool var_39;
    const wp::int32 var_40 = 0;
    wp::int32 var_41;
    //---------
    // forward
    // def remove_zero_length_edges(loop: wp.array[wp.vec2], loop_count: int, eps: float) -> int:       <L 584>
    // if loop_count < 2:                                                                     <L 596>
    var_1 = (var_loop_count < var_0);
    if (var_1) {
        // return 0                                                                           <L 597>
        return var_2;
    }
    // write_idx = int(0)                                                                     <L 599>
    var_4 = wp::int(var_3);
    // for read_idx in range(1, loop_count):                                                  <L 601>
    var_6 = wp::range(var_5, var_loop_count);
    start_for_1:;
        if (iter_cmp(var_6) == 0) goto end_for_1;
        var_7 = wp::iter_next(var_6);
        // diff = loop[read_idx] - loop[write_idx]                                            <L 602>
        var_8 = wp::address(var_loop, var_7);
        var_9 = wp::address(var_loop, var_4);
        var_11 = wp::load(var_8);
        var_12 = wp::load(var_9);
        var_10 = wp::sub(var_11, var_12);
        // if wp.length_sq(diff) > eps:                                                       <L 603>
        var_13 = wp::length_sq(var_10);
        var_14 = (var_13 > var_eps);
        if (var_14) {
            // write_idx += 1                                                                 <L 604>
            var_16 = wp::add(var_4, var_15);
            // loop[write_idx] = loop[read_idx]                                               <L 605>
            var_17 = wp::address(var_loop, var_7);
            var_18 = wp::load(var_17);
            wp::array_store(var_loop, var_16, var_18);
        }
        var_19 = wp::where(var_14, var_16, var_4);
        wp::assign(var_4, var_19);
        goto start_for_1;
    end_for_1:;
    // if write_idx > 0:                                                                      <L 608>
    var_21 = (var_4 > var_20);
    if (var_21) {
        // diff = loop[write_idx] - loop[0]                                                   <L 609>
        var_22 = wp::address(var_loop, var_4);
        var_24 = wp::address(var_loop, var_23);
        var_26 = wp::load(var_22);
        var_27 = wp::load(var_24);
        var_25 = wp::sub(var_26, var_27);
        // if wp.length_sq(diff) < eps:                                                       <L 610>
        var_28 = wp::length_sq(var_25);
        var_29 = (var_28 < var_eps);
        if (var_29) {
            // new_loop_count = write_idx                                                     <L 611>
            var_30 = wp::copy(var_4);
        }
        if (!var_29) {
            // new_loop_count = write_idx + 1                                                 <L 613>
            var_32 = wp::add(var_4, var_31);
        }
        var_33 = wp::where(var_29, var_30, var_32);
    }
    if (!var_21) {
        // new_loop_count = write_idx + 1                                                     <L 615>
        var_35 = wp::add(var_4, var_34);
    }
    var_36 = wp::where(var_21, var_25, var_10);
    var_37 = wp::where(var_21, var_33, var_35);
    // if new_loop_count < 2:                                                                 <L 617>
    var_39 = (var_37 < var_38);
    if (var_39) {
        // new_loop_count = 0                                                                 <L 618>
    }
    var_41 = wp::where(var_39, var_40, var_37);
    // return new_loop_count                                                                  <L 620>
    return var_41;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:487
static CUDA_CALLABLE wp::vec_t<4, wp::int32> approx_max_quadrilateral_area_with_calipers_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_hull,
    wp::int32 var_hull_count)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    const wp::int32 var_3 = 1;
    wp::int32 var_4;
    wp::vec_t<2, wp::float32>* var_5;
    wp::vec_t<2, wp::float32> var_6;
    wp::vec_t<2, wp::float32> var_7;
    wp::vec_t<2, wp::float32>* var_8;
    wp::vec_t<2, wp::float32> var_9;
    wp::vec_t<2, wp::float32> var_10;
    const wp::int32 var_11 = 0;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::vec_t<2, wp::float32> var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.001;
    const wp::int32 var_34 = 1;
    wp::int32 var_35;
    wp::range_t var_36;
    wp::int32 var_37;
    wp::vec_t<2, wp::float32>* var_38;
    wp::vec_t<2, wp::float32> var_39;
    wp::vec_t<2, wp::float32> var_40;
    const wp::int32 var_41 = 1;
    wp::int32 var_42;
    wp::int32 var_43;
    wp::vec_t<2, wp::float32>* var_44;
    wp::vec_t<2, wp::float32> var_45;
    wp::vec_t<2, wp::float32> var_46;
    const bool var_47 = true;
    wp::vec_t<2, wp::float32>* var_48;
    wp::vec_t<2, wp::float32> var_49;
    wp::vec_t<2, wp::float32> var_50;
    const wp::int32 var_51 = 1;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::vec_t<2, wp::float32>* var_54;
    wp::vec_t<2, wp::float32> var_55;
    wp::vec_t<2, wp::float32> var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    bool var_59;
    const wp::int32 var_60 = 1;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::vec_t<2, wp::float32>* var_64;
    wp::vec_t<2, wp::float32> var_65;
    wp::vec_t<2, wp::float32> var_66;
    wp::vec_t<2, wp::float32>* var_67;
    wp::vec_t<2, wp::float32> var_68;
    wp::vec_t<2, wp::float32> var_69;
    const wp::int32 var_70 = 0;
    wp::float32 var_71;
    const wp::int32 var_72 = 0;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 1;
    wp::float32 var_76;
    const wp::int32 var_77 = 1;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::vec_t<2, wp::float32> var_80;
    const wp::int32 var_81 = 0;
    wp::float32 var_82;
    const wp::int32 var_83 = 0;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 1;
    wp::float32 var_87;
    const wp::int32 var_88 = 1;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::float32 var_92 = 1.0;
    wp::float32 var_93;
    wp::float32 var_94;
    bool var_95;
    wp::float32 var_96;
    wp::int32 var_97;
    wp::int32 var_98;
    wp::int32 var_99;
    wp::int32 var_100;
    wp::float32 var_101;
    const wp::int32 var_102 = 1;
    wp::int32 var_103;
    wp::int32 var_104;
    wp::vec_t<2, wp::float32>* var_105;
    wp::vec_t<2, wp::float32> var_106;
    wp::vec_t<2, wp::float32> var_107;
    const wp::int32 var_108 = 0;
    wp::float32 var_109;
    const wp::int32 var_110 = 0;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::int32 var_113 = 1;
    wp::float32 var_114;
    const wp::int32 var_115 = 1;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::vec_t<2, wp::float32> var_118;
    const wp::int32 var_119 = 0;
    wp::float32 var_120;
    const wp::int32 var_121 = 0;
    wp::float32 var_122;
    wp::float32 var_123;
    const wp::int32 var_124 = 1;
    wp::float32 var_125;
    const wp::int32 var_126 = 1;
    wp::float32 var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    const wp::float32 var_130 = 1.0;
    wp::float32 var_131;
    wp::float32 var_132;
    bool var_133;
    wp::float32 var_134;
    const wp::int32 var_135 = 1;
    wp::int32 var_136;
    wp::int32 var_137;
    wp::int32 var_138;
    wp::int32 var_139;
    wp::int32 var_140;
    wp::float32 var_141;
    const wp::int32 var_142 = 0;
    wp::int32 var_143;
    const wp::int32 var_144 = 0;
    wp::int32 var_145;
    const wp::float32 var_146 = 0.0;
    wp::float32 var_147;
    const wp::float32 var_148 = 0.0;
    wp::float32 var_149;
    wp::vec_t<2, wp::float32>* var_150;
    wp::vec_t<2, wp::float32> var_151;
    wp::vec_t<2, wp::float32> var_152;
    wp::vec_t<2, wp::float32>* var_153;
    wp::vec_t<2, wp::float32> var_154;
    wp::vec_t<2, wp::float32> var_155;
    wp::range_t var_156;
    wp::int32 var_157;
    wp::vec_t<2, wp::float32>* var_158;
    wp::vec_t<2, wp::float32> var_159;
    wp::vec_t<2, wp::float32> var_160;
    wp::float32 var_161;
    const wp::float32 var_162 = 1.0;
    wp::float32 var_163;
    wp::float32 var_164;
    bool var_165;
    wp::float32 var_166;
    wp::int32 var_167;
    wp::float32 var_168;
    const wp::float32 var_169 = 1.0;
    wp::float32 var_170;
    wp::float32 var_171;
    bool var_172;
    wp::float32 var_173;
    wp::int32 var_174;
    wp::int32 var_175;
    wp::float32 var_176;
    wp::int32 var_177;
    wp::int32 var_178;
    wp::float32 var_179;
    wp::float32 var_180;
    wp::vec_t<4, wp::int32> var_181;
    //---------
    // forward
    // def approx_max_quadrilateral_area_with_calipers(hull: wp.array[wp.vec2], hull_count: int) -> wp.vec4i:       <L 488>
    // n = hull_count                                                                         <L 501>
    var_0 = wp::copy(var_hull_count);
    // p1 = int(0)                                                                            <L 504>
    var_2 = wp::int(var_1);
    // p3 = int(1)                                                                            <L 505>
    var_4 = wp::int(var_3);
    // hp1 = hull[p1]                                                                         <L 506>
    var_5 = wp::address(var_hull, var_2);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // hp3 = hull[p3]                                                                         <L 507>
    var_8 = wp::address(var_hull, var_4);
    var_10 = wp::load(var_8);
    var_9 = wp::copy(var_10);
    // diff = wp.vec2(hp1[0] - hp3[0], hp1[1] - hp3[1])                                       <L 508>
    var_12 = wp::extract(var_6, var_11);
    var_14 = wp::extract(var_9, var_13);
    var_15 = wp::sub(var_12, var_14);
    var_17 = wp::extract(var_6, var_16);
    var_19 = wp::extract(var_9, var_18);
    var_20 = wp::sub(var_17, var_19);
    var_21 = wp::vec_t<2, wp::float32>(var_15, var_20);
    // max_dist_sq = diff[0] * diff[0] + diff[1] * diff[1]                                    <L 509>
    var_23 = wp::extract(var_21, var_22);
    var_25 = wp::extract(var_21, var_24);
    var_26 = wp::mul(var_23, var_25);
    var_28 = wp::extract(var_21, var_27);
    var_30 = wp::extract(var_21, var_29);
    var_31 = wp::mul(var_28, var_30);
    var_32 = wp::add(var_26, var_31);
    // tie_epsilon_rel = 1.0e-3                                                               <L 514>
    // j = int(1)                                                                             <L 517>
    var_35 = wp::int(var_34);
    // for i in range(n):                                                                     <L 518>
    var_36 = wp::range(var_0);
    start_for_0:;
        if (iter_cmp(var_36) == 0) goto end_for_0;
        var_37 = wp::iter_next(var_36);
        // hull_i = hull[i]                                                                   <L 522>
        var_38 = wp::address(var_hull, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // hull_i_plus_1 = hull[(i + 1) % n]                                                  <L 523>
        var_42 = wp::add(var_37, var_41);
        var_43 = wp::mod(var_42, var_0);
        var_44 = wp::address(var_hull, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        // while True:                                                                        <L 525>
    start_while_2:;
    if ((var_47) == false) goto end_while_2;
            // hull_j = hull[j]                                                               <L 526>
            var_48 = wp::address(var_hull, var_35);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // hull_j_plus_1 = hull[(j + 1) % n]                                              <L 527>
            var_52 = wp::add(var_35, var_51);
            var_53 = wp::mod(var_52, var_0);
            var_54 = wp::address(var_hull, var_53);
            var_56 = wp::load(var_54);
            var_55 = wp::copy(var_56);
            // area_j_plus_1 = signed_area(hull_i, hull_i_plus_1, hull_j_plus_1)              <L 529>
            var_57 = signed_area_0(var_39, var_45, var_55);
            // area_j = signed_area(hull_i, hull_i_plus_1, hull_j)                            <L 530>
            var_58 = signed_area_0(var_39, var_45, var_49);
            // if area_j_plus_1 > area_j:                                                     <L 532>
            var_59 = (var_57 > var_58);
            if (var_59) {
                // j = (j + 1) % n                                                            <L 533>
                var_61 = wp::add(var_35, var_60);
                var_62 = wp::mod(var_61, var_0);
            }
            if (!var_59) {
                // break                                                                      <L 535>
                goto end_while_2;
            }
            var_63 = wp::where(var_59, var_62, var_35);
            wp::assign(var_35, var_63);
    goto start_while_2;
    end_while_2:;
        // hi = hull[i]                                                                       <L 538>
        var_64 = wp::address(var_hull, var_37);
        var_66 = wp::load(var_64);
        var_65 = wp::copy(var_66);
        // hj = hull[j]                                                                       <L 539>
        var_67 = wp::address(var_hull, var_35);
        var_69 = wp::load(var_67);
        var_68 = wp::copy(var_69);
        // d1 = wp.vec2(hi[0] - hj[0], hi[1] - hj[1])                                         <L 540>
        var_71 = wp::extract(var_65, var_70);
        var_73 = wp::extract(var_68, var_72);
        var_74 = wp::sub(var_71, var_73);
        var_76 = wp::extract(var_65, var_75);
        var_78 = wp::extract(var_68, var_77);
        var_79 = wp::sub(var_76, var_78);
        var_80 = wp::vec_t<2, wp::float32>(var_74, var_79);
        // dist_sq_1 = d1[0] * d1[0] + d1[1] * d1[1]                                          <L 541>
        var_82 = wp::extract(var_80, var_81);
        var_84 = wp::extract(var_80, var_83);
        var_85 = wp::mul(var_82, var_84);
        var_87 = wp::extract(var_80, var_86);
        var_89 = wp::extract(var_80, var_88);
        var_90 = wp::mul(var_87, var_89);
        var_91 = wp::add(var_85, var_90);
        // if dist_sq_1 > max_dist_sq * (1.0 + tie_epsilon_rel):                              <L 543>
        var_93 = wp::add(var_92, var_33);
        var_94 = wp::mul(var_32, var_93);
        var_95 = (var_91 > var_94);
        if (var_95) {
            // max_dist_sq = dist_sq_1                                                        <L 544>
            var_96 = wp::copy(var_91);
            // p1 = i                                                                         <L 545>
            var_97 = wp::copy(var_37);
            // p3 = j                                                                         <L 546>
            var_98 = wp::copy(var_35);
        }
        var_99 = wp::where(var_95, var_97, var_2);
        var_100 = wp::where(var_95, var_98, var_4);
        var_101 = wp::where(var_95, var_96, var_32);
        // hip1 = hull[(i + 1) % n]                                                           <L 549>
        var_103 = wp::add(var_37, var_102);
        var_104 = wp::mod(var_103, var_0);
        var_105 = wp::address(var_hull, var_104);
        var_107 = wp::load(var_105);
        var_106 = wp::copy(var_107);
        // d2 = wp.vec2(hip1[0] - hj[0], hip1[1] - hj[1])                                     <L 550>
        var_109 = wp::extract(var_106, var_108);
        var_111 = wp::extract(var_68, var_110);
        var_112 = wp::sub(var_109, var_111);
        var_114 = wp::extract(var_106, var_113);
        var_116 = wp::extract(var_68, var_115);
        var_117 = wp::sub(var_114, var_116);
        var_118 = wp::vec_t<2, wp::float32>(var_112, var_117);
        // dist_sq_2 = d2[0] * d2[0] + d2[1] * d2[1]                                          <L 551>
        var_120 = wp::extract(var_118, var_119);
        var_122 = wp::extract(var_118, var_121);
        var_123 = wp::mul(var_120, var_122);
        var_125 = wp::extract(var_118, var_124);
        var_127 = wp::extract(var_118, var_126);
        var_128 = wp::mul(var_125, var_127);
        var_129 = wp::add(var_123, var_128);
        // if dist_sq_2 > max_dist_sq * (1.0 + tie_epsilon_rel):                              <L 553>
        var_131 = wp::add(var_130, var_33);
        var_132 = wp::mul(var_101, var_131);
        var_133 = (var_129 > var_132);
        if (var_133) {
            // max_dist_sq = dist_sq_2                                                        <L 554>
            var_134 = wp::copy(var_129);
            // p1 = (i + 1) % n                                                               <L 555>
            var_136 = wp::add(var_37, var_135);
            var_137 = wp::mod(var_136, var_0);
            // p3 = j                                                                         <L 556>
            var_138 = wp::copy(var_35);
        }
        var_139 = wp::where(var_133, var_137, var_99);
        var_140 = wp::where(var_133, var_138, var_100);
        var_141 = wp::where(var_133, var_134, var_101);
        wp::assign(var_2, var_139);
        wp::assign(var_4, var_140);
        wp::assign(var_32, var_141);
        goto start_for_0;
    end_for_0:;
    // p2 = int(0)                                                                            <L 559>
    var_143 = wp::int(var_142);
    // p4 = int(0)                                                                            <L 560>
    var_145 = wp::int(var_144);
    // max_area_1 = float(0.0)                                                                <L 561>
    var_147 = wp::float(var_146);
    // max_area_2 = float(0.0)                                                                <L 562>
    var_149 = wp::float(var_148);
    // hull_p1 = hull[p1]                                                                     <L 564>
    var_150 = wp::address(var_hull, var_2);
    var_152 = wp::load(var_150);
    var_151 = wp::copy(var_152);
    // hull_p3 = hull[p3]                                                                     <L 565>
    var_153 = wp::address(var_hull, var_4);
    var_155 = wp::load(var_153);
    var_154 = wp::copy(var_155);
    // for i in range(n):                                                                     <L 567>
    var_156 = wp::range(var_0);
    start_for_4:;
        if (iter_cmp(var_156) == 0) goto end_for_4;
        var_157 = wp::iter_next(var_156);
        // hull_i = hull[i]                                                                   <L 569>
        var_158 = wp::address(var_hull, var_157);
        var_160 = wp::load(var_158);
        var_159 = wp::copy(var_160);
        // area = signed_area(hull_p1, hull_p3, hull_i)                                       <L 570>
        var_161 = signed_area_0(var_151, var_154, var_159);
        // if area > max_area_1 * (1.0 + tie_epsilon_rel):                                    <L 573>
        var_163 = wp::add(var_162, var_33);
        var_164 = wp::mul(var_147, var_163);
        var_165 = (var_161 > var_164);
        if (var_165) {
            // max_area_1 = area                                                              <L 574>
            var_166 = wp::copy(var_161);
            // p2 = i                                                                         <L 575>
            var_167 = wp::copy(var_157);
        }
        if (!var_165) {
            // elif -area > max_area_2 * (1.0 + tie_epsilon_rel):  # Check the other side       <L 576>
            var_168 = wp::neg(var_161);
            var_170 = wp::add(var_169, var_33);
            var_171 = wp::mul(var_149, var_170);
            var_172 = (var_168 > var_171);
            if (var_172) {
                // max_area_2 = -area                                                         <L 577>
                var_173 = wp::neg(var_161);
                // p4 = i                                                                     <L 578>
                var_174 = wp::copy(var_157);
            }
            var_175 = wp::where(var_172, var_174, var_145);
            var_176 = wp::where(var_172, var_173, var_149);
        }
        var_177 = wp::where(var_165, var_167, var_143);
        var_178 = wp::where(var_165, var_145, var_175);
        var_179 = wp::where(var_165, var_166, var_147);
        var_180 = wp::where(var_165, var_149, var_176);
        wp::assign(var_39, var_159);
        wp::assign(var_143, var_177);
        wp::assign(var_145, var_178);
        wp::assign(var_147, var_179);
        wp::assign(var_149, var_180);
        goto start_for_4;
    end_for_4:;
    // return wp.vec4i(p1, p2, p3, p4)                                                        <L 580>
    var_181 = wp::vec_t<4, wp::int32>(var_2, var_143, var_4, var_145);
    return var_181;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:87
static CUDA_CALLABLE wp::vec_t<3, wp::float32> ray_plane_intersection_0(
    wp::vec_t<3, wp::float32> var_ray_origin,
    wp::vec_t<3, wp::float32> var_ray_direction,
    wp::float32 var_plane_d,
    wp::vec_t<3, wp::float32> var_plane_normal)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 1e-12;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    //---------
    // forward
    // def ray_plane_intersection(                                                            <L 88>
    // denom = wp.dot(ray_direction, plane_normal)                                            <L 106>
    var_0 = wp::dot(var_ray_direction, var_plane_normal);
    // if wp.abs(denom) < 1.0e-12:                                                            <L 108>
    var_1 = wp::abs(var_0);
    var_3 = (var_1 < var_2);
    if (var_3) {
        // return ray_origin                                                                  <L 109>
        return var_ray_origin;
    }
    // t = -(wp.dot(ray_origin, plane_normal) + plane_d) / denom                              <L 113>
    var_4 = wp::dot(var_ray_origin, var_plane_normal);
    var_5 = wp::add(var_4, var_plane_d);
    var_6 = wp::neg(var_5);
    var_7 = wp::div(var_6, var_0);
    // return ray_origin + ray_direction * t                                                  <L 114>
    var_8 = wp::mul(var_ray_direction, var_7);
    var_9 = wp::add(var_ray_origin, var_8);
    return var_9;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:239
static CUDA_CALLABLE wp::vec_t<3, wp::float32> body_projector_project_0(
    BodyProjector_d067bb7a var_proj,
    wp::vec_t<3, wp::float32> var_input,
    wp::vec_t<3, wp::float32> var_contact_normal)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def body_projector_project(                                                            <L 240>
    // return ray_plane_intersection(input, contact_normal, proj.plane_d, proj.normal)        <L 260>
    var_0 = &((var_proj).plane_d);
    var_1 = &((var_proj).normal);
    var_3 = wp::load(var_0);
    var_4 = wp::load(var_1);
    var_2 = ray_plane_intersection_0(var_input, var_contact_normal, var_3, var_4);
    return var_2;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:0
static CUDA_CALLABLE void create_build_manifold__locals__extract_4_point_contact_manifolds_5(
    wp::array_t<wp::vec_t<2, wp::float32>> var_m_a,
    wp::int32 var_m_a_count,
    wp::fixedarray_t<10, wp::vec_t<2, wp::float32>> var_m_b,
    wp::int32 var_m_b_count,
    wp::vec_t<3, wp::float32> var_normal_local,
    wp::vec_t<3, wp::float32> var_cross_vector_1,
    wp::vec_t<3, wp::float32> var_cross_vector_2,
    wp::vec_t<3, wp::float32> var_center_local,
    BodyProjector_d067bb7a var_projector_a,
    BodyProjector_d067bb7a var_projector_b,
    wp::quat_t<wp::float32> var_orientation_a,
    wp::vec_t<3, wp::float32> var_position_a_world,
    wp::vec_t<3, wp::float32> var_normal_world,
    ContactWriterData_099b2624 var_writer_data,
    ContactData_40360d7c var_contact_template,
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_position_a,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::quat_t<wp::float32> var_quaternion_a,
    wp::quat_t<wp::float32> var_quaternion_b,
    wp::int32 & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::float32 var_5;
    wp::int32 var_6;
    const wp::float32 var_7 = 1e-05;
    wp::int32 var_8;
    const wp::int32 var_9 = 1;
    bool var_10;
    wp::vec_t<4, wp::int32> var_11;
    const wp::int32 var_12 = 4;
    bool var_13;
    wp::vec_t<4, wp::int32> var_14;
    const wp::int32 var_15 = 4;
    const wp::int32 var_16 = 0;
    const wp::int32 var_17 = 1;
    const wp::int32 var_18 = 2;
    const wp::int32 var_19 = 3;
    wp::vec_t<4, wp::int32> var_20;
    wp::int32 var_21;
    wp::vec_t<4, wp::int32> var_22;
    wp::range_t var_23;
    wp::int32 var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::vec_t<2, wp::float32>* var_27;
    const wp::int32 var_28 = 0;
    wp::float32 var_29;
    wp::vec_t<2, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<2, wp::float32>* var_32;
    const wp::int32 var_33 = 1;
    wp::float32 var_34;
    wp::vec_t<2, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    const wp::float32 var_41 = 0.5;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::float32 var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    ContactData_40360d7c var_48;
    wp::int32* var_49;
    const wp::int32 var_50 = 3;
    wp::int32 var_51;
    wp::int32 var_52;
    wp::int32 var_53;
    ContactData_40360d7c var_54;
    const wp::int32 var_55 = -1;
    const wp::float32 var_56 = 0.0;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    wp::int32 var_59;
    //---------
    // forward
    // def extract_4_point_contact_manifolds(                                                 <L 1>
    // normal_dot = wp.abs(wp.dot(projector_a.normal, projector_b.normal))                    <L 59>
    var_0 = &((var_projector_a).normal);
    var_1 = &((var_projector_b).normal);
    var_3 = wp::load(var_0);
    var_4 = wp::load(var_1);
    var_2 = wp::dot(var_3, var_4);
    var_5 = wp::abs(var_2);
    // loop_count = trim_all_in_place(m_a, m_a_count, m_b, m_b_count)                         <L 61>
    var_6 = trim_all_in_place_0(var_m_a, var_m_a_count, var_m_b, var_m_b_count);
    // loop_count = remove_zero_length_edges(m_b, loop_count, EPS)                            <L 63>
    var_8 = remove_zero_length_edges_0(var_m_b, var_6, var_7);
    // if loop_count > 1:                                                                     <L 65>
    var_10 = (var_8 > var_9);
    if (var_10) {
        // result = wp.vec4i()                                                                <L 66>
        var_11 = wp::vec_t<4, wp::int32>();
        // if loop_count > 4:                                                                 <L 67>
        var_13 = (var_8 > var_12);
        if (var_13) {
            // result = approx_max_quadrilateral_area_with_calipers(m_b, loop_count)          <L 68>
            var_14 = approx_max_quadrilateral_area_with_calipers_0(var_m_b, var_8);
            // loop_count = 4                                                                 <L 69>
        }
        if (!var_13) {
            // result = wp.vec4i(0, 1, 2, 3)                                                  <L 71>
            var_20 = wp::vec_t<4, wp::int32>(var_16, var_17, var_18, var_19);
        }
        var_21 = wp::where(var_13, var_15, var_8);
        var_22 = wp::where(var_13, var_14, var_20);
        // for i in range(loop_count):                                                        <L 73>
        var_23 = wp::range(var_21);
        start_for_0:;
            if (iter_cmp(var_23) == 0) goto end_for_0;
            var_24 = wp::iter_next(var_23);
            // ia = int(result[i])                                                            <L 74>
            var_25 = wp::extract(var_22, var_24);
            var_26 = wp::int(var_25);
            // p_local = m_b[ia].x * cross_vector_1 + m_b[ia].y * cross_vector_2 + center_local       <L 77>
            var_27 = wp::address(var_m_b, var_26);
            var_30 = wp::load(var_27);
            var_29 = wp::extract(var_30, var_28);
            var_31 = wp::mul(var_29, var_cross_vector_1);
            var_32 = wp::address(var_m_b, var_26);
            var_35 = wp::load(var_32);
            var_34 = wp::extract(var_35, var_33);
            var_36 = wp::mul(var_34, var_cross_vector_2);
            var_37 = wp::add(var_31, var_36);
            var_38 = wp::add(var_37, var_center_local);
            // a = body_projector_project(projector_a, p_local, normal_local)                 <L 79>
            var_39 = body_projector_project_0(var_projector_a, var_38, var_normal_local);
            // b = body_projector_project(projector_b, p_local, normal_local)                 <L 80>
            var_40 = body_projector_project_0(var_projector_b, var_38, var_normal_local);
            // contact_point_local = 0.5 * (a + b)                                            <L 81>
            var_42 = wp::add(var_39, var_40);
            var_43 = wp::mul(var_41, var_42);
            // signed_distance = wp.dot(b - a, normal_local)                                  <L 82>
            var_44 = wp::sub(var_40, var_39);
            var_45 = wp::dot(var_44, var_normal_local);
            // contact_point_world = wp.quat_rotate(orientation_a, contact_point_local) + position_a_world       <L 85>
            var_46 = wp::quat_rotate(var_orientation_a, var_43);
            var_47 = wp::add(var_46, var_position_a_world);
            // contact_data = contact_template                                                <L 87>
            var_48 = wp::copy(var_contact_template);
            // contact_data.contact_point_center = contact_point_world                        <L 88>
            var_48.contact_point_center = var_47;
            // contact_data.contact_normal_a_to_b = normal_world                              <L 89>
            var_48.contact_normal_a_to_b = var_normal_world;
            // contact_data.contact_distance = signed_distance                                <L 90>
            var_48.contact_distance = var_45;
            // contact_data.sort_sub_key = (contact_template.sort_sub_key << 3) | i           <L 91>
            var_49 = &((var_contact_template).sort_sub_key);
            var_52 = wp::load(var_49);
            var_51 = wp::lshift(var_52, var_50);
            var_53 = wp::bit_or(var_51, var_24);
            var_48.sort_sub_key = var_53;
            // contact_data = post_process_contact(                                           <L 93>
            // contact_data, geom_a, position_a, quaternion_a, geom_b, position_b, quaternion_b       <L 94>
            var_54 = post_process_minkowski_only_0(var_48, var_geom_a, var_position_a, var_quaternion_a, var_geom_b, var_position_b, var_quaternion_b);
            // writer_func(contact_data, writer_data, -1)                                     <L 96>
            write_contact_0(var_54, var_writer_data, var_55);
            goto start_for_0;
        end_for_0:;
    }
    if (!var_10) {
        // normal_dot = 0.0                                                                   <L 98>
        // loop_count = 0                                                                     <L 99>
    }
    var_58 = wp::where(var_10, var_5, var_56);
    var_59 = wp::where(var_10, var_21, var_57);
    // return loop_count, normal_dot                                                          <L 101>
    ret_0 = var_59;
    ret_1 = var_58;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:38
static CUDA_CALLABLE bool should_include_deepest_contact_0(
    wp::float32 var_normal_dot)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.9999984769132877;
    bool var_1;
    //---------
    // forward
    // def should_include_deepest_contact(normal_dot: float) -> bool:                         <L 39>
    // return normal_dot < COS_DEEPEST_CONTACT_THRESHOLD_ANGLE                                <L 40>
    var_1 = (var_normal_dot < var_0);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:0
static CUDA_CALLABLE wp::int32 create_build_manifold__locals__build_manifold_5(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_a,
    wp::vec_t<3, wp::float32> var_position_a_world,
    wp::quat_t<wp::float32> var_relative_orientation_b,
    wp::vec_t<3, wp::float32> var_relative_position_b,
    wp::vec_t<3, wp::float32> var_p_a,
    wp::vec_t<3, wp::float32> var_p_b,
    wp::vec_t<3, wp::float32> var_normal,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    ContactWriterData_099b2624 var_writer_data,
    ContactData_40360d7c var_contact_template)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1.0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.30901699437494745;
    const wp::float32 var_5 = 0.9510565162951535;
    const wp::float32 var_6 = -0.8090169943749473;
    const wp::float32 var_7 = 0.5877852522924732;
    const wp::float32 var_8 = -0.8090169943749476;
    const wp::float32 var_9 = -0.587785252292473;
    const wp::float32 var_10 = 0.30901699437494723;
    const wp::float32 var_11 = -0.9510565162951536;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    const wp::int32 var_14 = 0;
    wp::int32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    IncrementalPlaneTracker_8af6ef33 var_18;
    IncrementalPlaneTracker_8af6ef33 var_19;
    const wp::float32 var_20 = 0.5;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    const wp::int32 var_23 = 10;
    wp::tuple_t<wp::int32> var_24;
    wp::fixedarray_t<10, wp::vec_t<2, wp::float32>> var_25;
    const wp::int32 var_26 = 10;
    wp::uint64* var_27;
    const wp::int32 var_28 = 5;
    const wp::int32 var_29 = 8;
    wp::int32 var_30;
    wp::uint64 var_31;
    wp::uint64 var_32;
    wp::uint64 var_33;
    const wp::int32 var_34 = 5;
    wp::tuple_t<wp::int32> var_35;
    wp::array_t<wp::vec_t<2, wp::float32>> var_36;
    const wp::int32 var_37 = 5;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    bool var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 2;
    bool var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::int32 var_55 = 3;
    bool var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::int32 var_59 = 4;
    bool var_60;
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
    const wp::float32 var_71 = 0.9993908270190958;
    wp::float32 var_72;
    const wp::float32 var_73 = 0.03489949670250097;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::vec_t<2, wp::float32> var_85;
    const wp::float32 var_86 = 1e-05;
    wp::int32 var_87;
    bool var_88;
    const wp::int32 var_89 = 1;
    wp::int32 var_90;
    IncrementalPlaneTracker_8af6ef33 var_91;
    IncrementalPlaneTracker_8af6ef33 var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::vec_t<2, wp::float32> var_104;
    wp::int32 var_105;
    bool var_106;
    const wp::int32 var_107 = 1;
    wp::int32 var_108;
    IncrementalPlaneTracker_8af6ef33 var_109;
    IncrementalPlaneTracker_8af6ef33 var_110;
    const wp::int32 var_111 = 1;
    wp::float32 var_112;
    wp::float32 var_113;
    const wp::int32 var_114 = 1;
    bool var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::int32 var_118 = 2;
    bool var_119;
    wp::float32 var_120;
    wp::float32 var_121;
    const wp::int32 var_122 = 3;
    bool var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 4;
    bool var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    wp::float32 var_133;
    wp::float32 var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::float32 var_138;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::vec_t<3, wp::float32> var_141;
    wp::vec_t<3, wp::float32> var_142;
    wp::vec_t<3, wp::float32> var_143;
    wp::vec_t<3, wp::float32> var_144;
    wp::vec_t<3, wp::float32> var_145;
    wp::vec_t<3, wp::float32> var_146;
    wp::vec_t<3, wp::float32> var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    wp::vec_t<2, wp::float32> var_150;
    wp::int32 var_151;
    bool var_152;
    const wp::int32 var_153 = 1;
    wp::int32 var_154;
    IncrementalPlaneTracker_8af6ef33 var_155;
    IncrementalPlaneTracker_8af6ef33 var_156;
    wp::vec_t<3, wp::float32> var_157;
    wp::vec_t<3, wp::float32> var_158;
    wp::vec_t<3, wp::float32> var_159;
    wp::vec_t<3, wp::float32> var_160;
    wp::vec_t<3, wp::float32> var_161;
    wp::vec_t<3, wp::float32> var_162;
    wp::vec_t<3, wp::float32> var_163;
    wp::vec_t<3, wp::float32> var_164;
    wp::vec_t<3, wp::float32> var_165;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::vec_t<2, wp::float32> var_168;
    wp::int32 var_169;
    bool var_170;
    const wp::int32 var_171 = 1;
    wp::int32 var_172;
    IncrementalPlaneTracker_8af6ef33 var_173;
    IncrementalPlaneTracker_8af6ef33 var_174;
    const wp::int32 var_175 = 2;
    wp::float32 var_176;
    wp::float32 var_177;
    const wp::int32 var_178 = 1;
    bool var_179;
    wp::float32 var_180;
    wp::float32 var_181;
    const wp::int32 var_182 = 2;
    bool var_183;
    wp::float32 var_184;
    wp::float32 var_185;
    const wp::int32 var_186 = 3;
    bool var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    const wp::int32 var_190 = 4;
    bool var_191;
    wp::float32 var_192;
    wp::float32 var_193;
    wp::float32 var_194;
    wp::float32 var_195;
    wp::float32 var_196;
    wp::float32 var_197;
    wp::float32 var_198;
    wp::float32 var_199;
    wp::float32 var_200;
    wp::float32 var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    wp::float32 var_204;
    wp::vec_t<3, wp::float32> var_205;
    wp::vec_t<3, wp::float32> var_206;
    wp::vec_t<3, wp::float32> var_207;
    wp::vec_t<3, wp::float32> var_208;
    wp::vec_t<3, wp::float32> var_209;
    wp::vec_t<3, wp::float32> var_210;
    wp::vec_t<3, wp::float32> var_211;
    wp::float32 var_212;
    wp::float32 var_213;
    wp::vec_t<2, wp::float32> var_214;
    wp::int32 var_215;
    bool var_216;
    const wp::int32 var_217 = 1;
    wp::int32 var_218;
    IncrementalPlaneTracker_8af6ef33 var_219;
    IncrementalPlaneTracker_8af6ef33 var_220;
    wp::vec_t<3, wp::float32> var_221;
    wp::vec_t<3, wp::float32> var_222;
    wp::vec_t<3, wp::float32> var_223;
    wp::vec_t<3, wp::float32> var_224;
    wp::vec_t<3, wp::float32> var_225;
    wp::vec_t<3, wp::float32> var_226;
    wp::vec_t<3, wp::float32> var_227;
    wp::vec_t<3, wp::float32> var_228;
    wp::vec_t<3, wp::float32> var_229;
    wp::float32 var_230;
    wp::float32 var_231;
    wp::vec_t<2, wp::float32> var_232;
    wp::int32 var_233;
    bool var_234;
    const wp::int32 var_235 = 1;
    wp::int32 var_236;
    IncrementalPlaneTracker_8af6ef33 var_237;
    IncrementalPlaneTracker_8af6ef33 var_238;
    const wp::int32 var_239 = 3;
    wp::float32 var_240;
    wp::float32 var_241;
    const wp::int32 var_242 = 1;
    bool var_243;
    wp::float32 var_244;
    wp::float32 var_245;
    const wp::int32 var_246 = 2;
    bool var_247;
    wp::float32 var_248;
    wp::float32 var_249;
    const wp::int32 var_250 = 3;
    bool var_251;
    wp::float32 var_252;
    wp::float32 var_253;
    const wp::int32 var_254 = 4;
    bool var_255;
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
    wp::float32 var_268;
    wp::vec_t<3, wp::float32> var_269;
    wp::vec_t<3, wp::float32> var_270;
    wp::vec_t<3, wp::float32> var_271;
    wp::vec_t<3, wp::float32> var_272;
    wp::vec_t<3, wp::float32> var_273;
    wp::vec_t<3, wp::float32> var_274;
    wp::vec_t<3, wp::float32> var_275;
    wp::float32 var_276;
    wp::float32 var_277;
    wp::vec_t<2, wp::float32> var_278;
    wp::int32 var_279;
    bool var_280;
    const wp::int32 var_281 = 1;
    wp::int32 var_282;
    IncrementalPlaneTracker_8af6ef33 var_283;
    IncrementalPlaneTracker_8af6ef33 var_284;
    wp::vec_t<3, wp::float32> var_285;
    wp::vec_t<3, wp::float32> var_286;
    wp::vec_t<3, wp::float32> var_287;
    wp::vec_t<3, wp::float32> var_288;
    wp::vec_t<3, wp::float32> var_289;
    wp::vec_t<3, wp::float32> var_290;
    wp::vec_t<3, wp::float32> var_291;
    wp::vec_t<3, wp::float32> var_292;
    wp::vec_t<3, wp::float32> var_293;
    wp::float32 var_294;
    wp::float32 var_295;
    wp::vec_t<2, wp::float32> var_296;
    wp::int32 var_297;
    bool var_298;
    const wp::int32 var_299 = 1;
    wp::int32 var_300;
    IncrementalPlaneTracker_8af6ef33 var_301;
    IncrementalPlaneTracker_8af6ef33 var_302;
    const wp::int32 var_303 = 4;
    wp::float32 var_304;
    wp::float32 var_305;
    const wp::int32 var_306 = 1;
    bool var_307;
    wp::float32 var_308;
    wp::float32 var_309;
    const wp::int32 var_310 = 2;
    bool var_311;
    wp::float32 var_312;
    wp::float32 var_313;
    const wp::int32 var_314 = 3;
    bool var_315;
    wp::float32 var_316;
    wp::float32 var_317;
    const wp::int32 var_318 = 4;
    bool var_319;
    wp::float32 var_320;
    wp::float32 var_321;
    wp::float32 var_322;
    wp::float32 var_323;
    wp::float32 var_324;
    wp::float32 var_325;
    wp::float32 var_326;
    wp::float32 var_327;
    wp::float32 var_328;
    wp::float32 var_329;
    wp::float32 var_330;
    wp::float32 var_331;
    wp::float32 var_332;
    wp::vec_t<3, wp::float32> var_333;
    wp::vec_t<3, wp::float32> var_334;
    wp::vec_t<3, wp::float32> var_335;
    wp::vec_t<3, wp::float32> var_336;
    wp::vec_t<3, wp::float32> var_337;
    wp::vec_t<3, wp::float32> var_338;
    wp::vec_t<3, wp::float32> var_339;
    wp::float32 var_340;
    wp::float32 var_341;
    wp::vec_t<2, wp::float32> var_342;
    wp::int32 var_343;
    bool var_344;
    const wp::int32 var_345 = 1;
    wp::int32 var_346;
    IncrementalPlaneTracker_8af6ef33 var_347;
    IncrementalPlaneTracker_8af6ef33 var_348;
    wp::vec_t<3, wp::float32> var_349;
    wp::vec_t<3, wp::float32> var_350;
    wp::vec_t<3, wp::float32> var_351;
    wp::vec_t<3, wp::float32> var_352;
    wp::vec_t<3, wp::float32> var_353;
    wp::vec_t<3, wp::float32> var_354;
    wp::vec_t<3, wp::float32> var_355;
    wp::vec_t<3, wp::float32> var_356;
    wp::vec_t<3, wp::float32> var_357;
    wp::float32 var_358;
    wp::float32 var_359;
    wp::vec_t<2, wp::float32> var_360;
    wp::int32 var_361;
    bool var_362;
    const wp::int32 var_363 = 1;
    wp::int32 var_364;
    IncrementalPlaneTracker_8af6ef33 var_365;
    IncrementalPlaneTracker_8af6ef33 var_366;
    wp::vec_t<3, wp::float32> var_367;
    wp::vec_t<3, wp::float32> var_368;
    wp::vec_t<3, wp::float32> var_369;
    wp::vec_t<3, wp::float32> var_370;
    wp::quat_t<wp::float32> var_371;
    wp::quat_t<wp::float32> var_372;
    bool var_373;
    const wp::int32 var_374 = 2;
    bool var_375;
    const wp::int32 var_376 = 2;
    bool var_377;
    const wp::int32 var_378 = 0;
    const wp::float32 var_379 = 0.0;
    BodyProjector_d067bb7a var_380;
    BodyProjector_d067bb7a var_381;
    bool var_382;
    wp::vec_t<3, wp::float32>* var_383;
    bool var_384;
    wp::vec_t<3, wp::float32> var_385;
    wp::vec_t<3, wp::float32>* var_386;
    bool var_387;
    wp::vec_t<3, wp::float32> var_388;
    const wp::int32 var_389 = 0;
    const wp::float32 var_390 = 0.0;
    wp::int32 var_391;
    wp::float32 var_392;
    const wp::int32 var_393 = 4;
    wp::int32 var_394;
    wp::int32 var_395;
    wp::float32 var_396;
    wp::int32 var_397;
    wp::float32 var_398;
    bool var_399;
    bool var_400;
    const wp::int32 var_401 = 0;
    bool var_402;
    const wp::float32 var_403 = 0.5;
    wp::vec_t<3, wp::float32> var_404;
    wp::vec_t<3, wp::float32> var_405;
    wp::vec_t<3, wp::float32> var_406;
    wp::float32 var_407;
    wp::vec_t<3, wp::float32> var_408;
    wp::vec_t<3, wp::float32> var_409;
    ContactData_40360d7c var_410;
    wp::int32* var_411;
    const wp::int32 var_412 = 3;
    wp::int32 var_413;
    wp::int32 var_414;
    wp::int32 var_415;
    ContactData_40360d7c var_416;
    const wp::int32 var_417 = -1;
    const wp::int32 var_418 = 1;
    wp::int32 var_419;
    wp::int32 var_420;
    //---------
    // forward
    // def build_manifold(                                                                    <L 1>
    // PENT_COS_0 = float(1.0)                                                                <L 41>
    var_1 = wp::float(var_0);
    // PENT_SIN_0 = float(0.0)                                                                <L 42>
    var_3 = wp::float(var_2);
    // PENT_COS_1 = wp.static(math.cos(2.0 * math.pi / 5.0))                                  <L 43>
    // PENT_SIN_1 = wp.static(math.sin(2.0 * math.pi / 5.0))                                  <L 44>
    // PENT_COS_2 = wp.static(math.cos(4.0 * math.pi / 5.0))                                  <L 45>
    // PENT_SIN_2 = wp.static(math.sin(4.0 * math.pi / 5.0))                                  <L 46>
    // PENT_COS_3 = wp.static(math.cos(6.0 * math.pi / 5.0))                                  <L 47>
    // PENT_SIN_3 = wp.static(math.sin(6.0 * math.pi / 5.0))                                  <L 48>
    // PENT_COS_4 = wp.static(math.cos(8.0 * math.pi / 5.0))                                  <L 49>
    // PENT_SIN_4 = wp.static(math.sin(8.0 * math.pi / 5.0))                                  <L 50>
    // a_count = int(0)                                                                       <L 52>
    var_13 = wp::int(var_12);
    // b_count = int(0)                                                                       <L 53>
    var_15 = wp::int(var_14);
    // tangent_a, tangent_b = orthonormal_basis(normal)                                       <L 56>
    orthonormal_basis_0(var_normal, var_16, var_17);
    // plane_tracker_a = IncrementalPlaneTracker()                                            <L 58>
    var_18 = IncrementalPlaneTracker_8af6ef33();
    // plane_tracker_b = IncrementalPlaneTracker()                                            <L 59>
    var_19 = IncrementalPlaneTracker_8af6ef33();
    // center = 0.5 * (p_a + p_b)                                                             <L 61>
    var_21 = wp::add(var_p_a, var_p_b);
    var_22 = wp::mul(var_20, var_21);
    // b_buffer = wp.zeros(shape=(10,), dtype=wp.vec2f)                                       <L 64>
    var_24 = wp::tuple(var_23);
    var_25 = wp::fixedarray_t<10, wp::vec_t<2, wp::float32>>(var_26);
    // a_buffer = wp.array(ptr=b_buffer.ptr + wp.uint64(5 * 8), shape=(5,), dtype=wp.vec2f)       <L 65>
    var_27 = (wp::uint64*)&(var_25.data);
    var_30 = wp::mul(var_28, var_29);
    var_31 = wp::uint64(var_30);
    var_33 = wp::load(var_27);
    var_32 = wp::add(var_33, var_31);
    var_35 = wp::tuple(var_34);
    var_36 = wp::array_t<wp::vec_t<2, wp::float32>>(var_32, var_37);
    // local_normal_b = wp.quat_rotate_inv(relative_orientation_b, -normal)                   <L 70>
    var_38 = wp::neg(var_normal);
    var_39 = wp::quat_rotate_inv(var_relative_orientation_b, var_38);
    // local_ta_b = wp.quat_rotate_inv(relative_orientation_b, -tangent_a)                    <L 71>
    var_40 = wp::neg(var_16);
    var_41 = wp::quat_rotate_inv(var_relative_orientation_b, var_40);
    // local_tb_b = wp.quat_rotate_inv(relative_orientation_b, -tangent_b)                    <L 72>
    var_42 = wp::neg(var_17);
    var_43 = wp::quat_rotate_inv(var_relative_orientation_b, var_42);
    // for e in range(5):                                                                     <L 74>
    // c = PENT_COS_0                                                                         <L 75>
    var_45 = wp::copy(var_1);
    // s = PENT_SIN_0                                                                         <L 76>
    var_46 = wp::copy(var_3);
    // if e == 1:                                                                             <L 77>
    var_48 = (var_44 == var_47);
    if (var_48) {
        // c = PENT_COS_1                                                                     <L 78>
        var_49 = wp::copy(var_4);
        // s = PENT_SIN_1                                                                     <L 79>
        var_50 = wp::copy(var_5);
    }
    if (!var_48) {
        // elif e == 2:                                                                       <L 80>
        var_52 = (var_44 == var_51);
        if (var_52) {
            // c = PENT_COS_2                                                                 <L 81>
            var_53 = wp::copy(var_6);
            // s = PENT_SIN_2                                                                 <L 82>
            var_54 = wp::copy(var_7);
        }
        if (!var_52) {
            // elif e == 3:                                                                   <L 83>
            var_56 = (var_44 == var_55);
            if (var_56) {
                // c = PENT_COS_3                                                             <L 84>
                var_57 = wp::copy(var_8);
                // s = PENT_SIN_3                                                             <L 85>
                var_58 = wp::copy(var_9);
            }
            if (!var_56) {
                // elif e == 4:                                                               <L 86>
                var_60 = (var_44 == var_59);
                if (var_60) {
                    // c = PENT_COS_4                                                         <L 87>
                    var_61 = wp::copy(var_10);
                    // s = PENT_SIN_4                                                         <L 88>
                    var_62 = wp::copy(var_11);
                }
                var_63 = wp::where(var_60, var_61, var_45);
                var_64 = wp::where(var_60, var_62, var_46);
            }
            var_65 = wp::where(var_56, var_57, var_63);
            var_66 = wp::where(var_56, var_58, var_64);
        }
        var_67 = wp::where(var_52, var_53, var_65);
        var_68 = wp::where(var_52, var_54, var_66);
    }
    var_69 = wp::where(var_48, var_49, var_67);
    var_70 = wp::where(var_48, var_50, var_68);
    // cos_tilt = COS_TILT_ANGLE                                                              <L 90>
    var_72 = wp::copy(var_71);
    // c_sin = c * SIN_TILT_ANGLE                                                             <L 91>
    var_74 = wp::mul(var_69, var_73);
    // s_sin = s * SIN_TILT_ANGLE                                                             <L 92>
    var_75 = wp::mul(var_70, var_73);
    // dir_a = normal * cos_tilt + c_sin * tangent_a + s_sin * tangent_b                      <L 95>
    var_76 = wp::mul(var_normal, var_72);
    var_77 = wp::mul(var_74, var_16);
    var_78 = wp::add(var_76, var_77);
    var_79 = wp::mul(var_75, var_17);
    var_80 = wp::add(var_78, var_79);
    // pt_a_3d = support_func(geom_a, dir_a, data_provider)                                   <L 96>
    var_81 = support_map_lean_0(var_geom_a, var_80, var_data_provider);
    // projected_a = pt_a_3d - center                                                         <L 97>
    var_82 = wp::sub(var_81, var_22);
    // pt_a_2d = wp.vec2(wp.dot(tangent_a, projected_a), wp.dot(tangent_b, projected_a))       <L 98>
    var_83 = wp::dot(var_16, var_82);
    var_84 = wp::dot(var_17, var_82);
    var_85 = wp::vec_t<2, wp::float32>(var_83, var_84);
    // a_count, was_added_a = add_avoid_duplicates_vec2(a_buffer, a_count, pt_a_2d, EPS)       <L 99>
    add_avoid_duplicates_vec2_0(var_36, var_13, var_85, var_86, var_87, var_88);
    // if was_added_a:                                                                        <L 100>
    if (var_88) {
        // plane_tracker_a = update_incremental_plane_tracker(plane_tracker_a, pt_a_3d, a_count - 1)       <L 101>
        var_90 = wp::sub(var_87, var_89);
        var_91 = update_incremental_plane_tracker_0(var_18, var_81, var_90);
    }
    var_92 = wp::where(var_88, var_91, var_18);
    // local_dir_b = local_normal_b * cos_tilt + c_sin * local_ta_b + s_sin * local_tb_b       <L 104>
    var_93 = wp::mul(var_39, var_72);
    var_94 = wp::mul(var_74, var_41);
    var_95 = wp::add(var_93, var_94);
    var_96 = wp::mul(var_75, var_43);
    var_97 = wp::add(var_95, var_96);
    // pt_b_local = support_func(geom_b, local_dir_b, data_provider)                          <L 105>
    var_98 = support_map_lean_0(var_geom_b, var_97, var_data_provider);
    // pt_b_3d = wp.quat_rotate(relative_orientation_b, pt_b_local) + relative_position_b       <L 106>
    var_99 = wp::quat_rotate(var_relative_orientation_b, var_98);
    var_100 = wp::add(var_99, var_relative_position_b);
    // projected_b = pt_b_3d - center                                                         <L 107>
    var_101 = wp::sub(var_100, var_22);
    // pt_b_2d = wp.vec2(wp.dot(tangent_a, projected_b), wp.dot(tangent_b, projected_b))       <L 108>
    var_102 = wp::dot(var_16, var_101);
    var_103 = wp::dot(var_17, var_101);
    var_104 = wp::vec_t<2, wp::float32>(var_102, var_103);
    // b_count, was_added_b = add_avoid_duplicates_vec2(b_buffer, b_count, pt_b_2d, EPS)       <L 109>
    add_avoid_duplicates_vec2_0(var_25, var_15, var_104, var_86, var_105, var_106);
    // if was_added_b:                                                                        <L 110>
    if (var_106) {
        // plane_tracker_b = update_incremental_plane_tracker(plane_tracker_b, pt_b_3d, b_count - 1)       <L 111>
        var_108 = wp::sub(var_105, var_107);
        var_109 = update_incremental_plane_tracker_0(var_19, var_100, var_108);
    }
    var_110 = wp::where(var_106, var_109, var_19);
    // c = PENT_COS_0                                                                         <L 75>
    var_112 = wp::copy(var_1);
    // s = PENT_SIN_0                                                                         <L 76>
    var_113 = wp::copy(var_3);
    // if e == 1:                                                                             <L 77>
    var_115 = (var_111 == var_114);
    if (var_115) {
        // c = PENT_COS_1                                                                     <L 78>
        var_116 = wp::copy(var_4);
        // s = PENT_SIN_1                                                                     <L 79>
        var_117 = wp::copy(var_5);
    }
    if (!var_115) {
        // elif e == 2:                                                                       <L 80>
        var_119 = (var_111 == var_118);
        if (var_119) {
            // c = PENT_COS_2                                                                 <L 81>
            var_120 = wp::copy(var_6);
            // s = PENT_SIN_2                                                                 <L 82>
            var_121 = wp::copy(var_7);
        }
        if (!var_119) {
            // elif e == 3:                                                                   <L 83>
            var_123 = (var_111 == var_122);
            if (var_123) {
                // c = PENT_COS_3                                                             <L 84>
                var_124 = wp::copy(var_8);
                // s = PENT_SIN_3                                                             <L 85>
                var_125 = wp::copy(var_9);
            }
            if (!var_123) {
                // elif e == 4:                                                               <L 86>
                var_127 = (var_111 == var_126);
                if (var_127) {
                    // c = PENT_COS_4                                                         <L 87>
                    var_128 = wp::copy(var_10);
                    // s = PENT_SIN_4                                                         <L 88>
                    var_129 = wp::copy(var_11);
                }
                var_130 = wp::where(var_127, var_128, var_112);
                var_131 = wp::where(var_127, var_129, var_113);
            }
            var_132 = wp::where(var_123, var_124, var_130);
            var_133 = wp::where(var_123, var_125, var_131);
        }
        var_134 = wp::where(var_119, var_120, var_132);
        var_135 = wp::where(var_119, var_121, var_133);
    }
    var_136 = wp::where(var_115, var_116, var_134);
    var_137 = wp::where(var_115, var_117, var_135);
    // cos_tilt = COS_TILT_ANGLE                                                              <L 90>
    var_138 = wp::copy(var_71);
    // c_sin = c * SIN_TILT_ANGLE                                                             <L 91>
    var_139 = wp::mul(var_136, var_73);
    // s_sin = s * SIN_TILT_ANGLE                                                             <L 92>
    var_140 = wp::mul(var_137, var_73);
    // dir_a = normal * cos_tilt + c_sin * tangent_a + s_sin * tangent_b                      <L 95>
    var_141 = wp::mul(var_normal, var_138);
    var_142 = wp::mul(var_139, var_16);
    var_143 = wp::add(var_141, var_142);
    var_144 = wp::mul(var_140, var_17);
    var_145 = wp::add(var_143, var_144);
    // pt_a_3d = support_func(geom_a, dir_a, data_provider)                                   <L 96>
    var_146 = support_map_lean_0(var_geom_a, var_145, var_data_provider);
    // projected_a = pt_a_3d - center                                                         <L 97>
    var_147 = wp::sub(var_146, var_22);
    // pt_a_2d = wp.vec2(wp.dot(tangent_a, projected_a), wp.dot(tangent_b, projected_a))       <L 98>
    var_148 = wp::dot(var_16, var_147);
    var_149 = wp::dot(var_17, var_147);
    var_150 = wp::vec_t<2, wp::float32>(var_148, var_149);
    // a_count, was_added_a = add_avoid_duplicates_vec2(a_buffer, a_count, pt_a_2d, EPS)       <L 99>
    add_avoid_duplicates_vec2_0(var_36, var_87, var_150, var_86, var_151, var_152);
    // if was_added_a:                                                                        <L 100>
    if (var_152) {
        // plane_tracker_a = update_incremental_plane_tracker(plane_tracker_a, pt_a_3d, a_count - 1)       <L 101>
        var_154 = wp::sub(var_151, var_153);
        var_155 = update_incremental_plane_tracker_0(var_92, var_146, var_154);
    }
    var_156 = wp::where(var_152, var_155, var_92);
    // local_dir_b = local_normal_b * cos_tilt + c_sin * local_ta_b + s_sin * local_tb_b       <L 104>
    var_157 = wp::mul(var_39, var_138);
    var_158 = wp::mul(var_139, var_41);
    var_159 = wp::add(var_157, var_158);
    var_160 = wp::mul(var_140, var_43);
    var_161 = wp::add(var_159, var_160);
    // pt_b_local = support_func(geom_b, local_dir_b, data_provider)                          <L 105>
    var_162 = support_map_lean_0(var_geom_b, var_161, var_data_provider);
    // pt_b_3d = wp.quat_rotate(relative_orientation_b, pt_b_local) + relative_position_b       <L 106>
    var_163 = wp::quat_rotate(var_relative_orientation_b, var_162);
    var_164 = wp::add(var_163, var_relative_position_b);
    // projected_b = pt_b_3d - center                                                         <L 107>
    var_165 = wp::sub(var_164, var_22);
    // pt_b_2d = wp.vec2(wp.dot(tangent_a, projected_b), wp.dot(tangent_b, projected_b))       <L 108>
    var_166 = wp::dot(var_16, var_165);
    var_167 = wp::dot(var_17, var_165);
    var_168 = wp::vec_t<2, wp::float32>(var_166, var_167);
    // b_count, was_added_b = add_avoid_duplicates_vec2(b_buffer, b_count, pt_b_2d, EPS)       <L 109>
    add_avoid_duplicates_vec2_0(var_25, var_105, var_168, var_86, var_169, var_170);
    // if was_added_b:                                                                        <L 110>
    if (var_170) {
        // plane_tracker_b = update_incremental_plane_tracker(plane_tracker_b, pt_b_3d, b_count - 1)       <L 111>
        var_172 = wp::sub(var_169, var_171);
        var_173 = update_incremental_plane_tracker_0(var_110, var_164, var_172);
    }
    var_174 = wp::where(var_170, var_173, var_110);
    // c = PENT_COS_0                                                                         <L 75>
    var_176 = wp::copy(var_1);
    // s = PENT_SIN_0                                                                         <L 76>
    var_177 = wp::copy(var_3);
    // if e == 1:                                                                             <L 77>
    var_179 = (var_175 == var_178);
    if (var_179) {
        // c = PENT_COS_1                                                                     <L 78>
        var_180 = wp::copy(var_4);
        // s = PENT_SIN_1                                                                     <L 79>
        var_181 = wp::copy(var_5);
    }
    if (!var_179) {
        // elif e == 2:                                                                       <L 80>
        var_183 = (var_175 == var_182);
        if (var_183) {
            // c = PENT_COS_2                                                                 <L 81>
            var_184 = wp::copy(var_6);
            // s = PENT_SIN_2                                                                 <L 82>
            var_185 = wp::copy(var_7);
        }
        if (!var_183) {
            // elif e == 3:                                                                   <L 83>
            var_187 = (var_175 == var_186);
            if (var_187) {
                // c = PENT_COS_3                                                             <L 84>
                var_188 = wp::copy(var_8);
                // s = PENT_SIN_3                                                             <L 85>
                var_189 = wp::copy(var_9);
            }
            if (!var_187) {
                // elif e == 4:                                                               <L 86>
                var_191 = (var_175 == var_190);
                if (var_191) {
                    // c = PENT_COS_4                                                         <L 87>
                    var_192 = wp::copy(var_10);
                    // s = PENT_SIN_4                                                         <L 88>
                    var_193 = wp::copy(var_11);
                }
                var_194 = wp::where(var_191, var_192, var_176);
                var_195 = wp::where(var_191, var_193, var_177);
            }
            var_196 = wp::where(var_187, var_188, var_194);
            var_197 = wp::where(var_187, var_189, var_195);
        }
        var_198 = wp::where(var_183, var_184, var_196);
        var_199 = wp::where(var_183, var_185, var_197);
    }
    var_200 = wp::where(var_179, var_180, var_198);
    var_201 = wp::where(var_179, var_181, var_199);
    // cos_tilt = COS_TILT_ANGLE                                                              <L 90>
    var_202 = wp::copy(var_71);
    // c_sin = c * SIN_TILT_ANGLE                                                             <L 91>
    var_203 = wp::mul(var_200, var_73);
    // s_sin = s * SIN_TILT_ANGLE                                                             <L 92>
    var_204 = wp::mul(var_201, var_73);
    // dir_a = normal * cos_tilt + c_sin * tangent_a + s_sin * tangent_b                      <L 95>
    var_205 = wp::mul(var_normal, var_202);
    var_206 = wp::mul(var_203, var_16);
    var_207 = wp::add(var_205, var_206);
    var_208 = wp::mul(var_204, var_17);
    var_209 = wp::add(var_207, var_208);
    // pt_a_3d = support_func(geom_a, dir_a, data_provider)                                   <L 96>
    var_210 = support_map_lean_0(var_geom_a, var_209, var_data_provider);
    // projected_a = pt_a_3d - center                                                         <L 97>
    var_211 = wp::sub(var_210, var_22);
    // pt_a_2d = wp.vec2(wp.dot(tangent_a, projected_a), wp.dot(tangent_b, projected_a))       <L 98>
    var_212 = wp::dot(var_16, var_211);
    var_213 = wp::dot(var_17, var_211);
    var_214 = wp::vec_t<2, wp::float32>(var_212, var_213);
    // a_count, was_added_a = add_avoid_duplicates_vec2(a_buffer, a_count, pt_a_2d, EPS)       <L 99>
    add_avoid_duplicates_vec2_0(var_36, var_151, var_214, var_86, var_215, var_216);
    // if was_added_a:                                                                        <L 100>
    if (var_216) {
        // plane_tracker_a = update_incremental_plane_tracker(plane_tracker_a, pt_a_3d, a_count - 1)       <L 101>
        var_218 = wp::sub(var_215, var_217);
        var_219 = update_incremental_plane_tracker_0(var_156, var_210, var_218);
    }
    var_220 = wp::where(var_216, var_219, var_156);
    // local_dir_b = local_normal_b * cos_tilt + c_sin * local_ta_b + s_sin * local_tb_b       <L 104>
    var_221 = wp::mul(var_39, var_202);
    var_222 = wp::mul(var_203, var_41);
    var_223 = wp::add(var_221, var_222);
    var_224 = wp::mul(var_204, var_43);
    var_225 = wp::add(var_223, var_224);
    // pt_b_local = support_func(geom_b, local_dir_b, data_provider)                          <L 105>
    var_226 = support_map_lean_0(var_geom_b, var_225, var_data_provider);
    // pt_b_3d = wp.quat_rotate(relative_orientation_b, pt_b_local) + relative_position_b       <L 106>
    var_227 = wp::quat_rotate(var_relative_orientation_b, var_226);
    var_228 = wp::add(var_227, var_relative_position_b);
    // projected_b = pt_b_3d - center                                                         <L 107>
    var_229 = wp::sub(var_228, var_22);
    // pt_b_2d = wp.vec2(wp.dot(tangent_a, projected_b), wp.dot(tangent_b, projected_b))       <L 108>
    var_230 = wp::dot(var_16, var_229);
    var_231 = wp::dot(var_17, var_229);
    var_232 = wp::vec_t<2, wp::float32>(var_230, var_231);
    // b_count, was_added_b = add_avoid_duplicates_vec2(b_buffer, b_count, pt_b_2d, EPS)       <L 109>
    add_avoid_duplicates_vec2_0(var_25, var_169, var_232, var_86, var_233, var_234);
    // if was_added_b:                                                                        <L 110>
    if (var_234) {
        // plane_tracker_b = update_incremental_plane_tracker(plane_tracker_b, pt_b_3d, b_count - 1)       <L 111>
        var_236 = wp::sub(var_233, var_235);
        var_237 = update_incremental_plane_tracker_0(var_174, var_228, var_236);
    }
    var_238 = wp::where(var_234, var_237, var_174);
    // c = PENT_COS_0                                                                         <L 75>
    var_240 = wp::copy(var_1);
    // s = PENT_SIN_0                                                                         <L 76>
    var_241 = wp::copy(var_3);
    // if e == 1:                                                                             <L 77>
    var_243 = (var_239 == var_242);
    if (var_243) {
        // c = PENT_COS_1                                                                     <L 78>
        var_244 = wp::copy(var_4);
        // s = PENT_SIN_1                                                                     <L 79>
        var_245 = wp::copy(var_5);
    }
    if (!var_243) {
        // elif e == 2:                                                                       <L 80>
        var_247 = (var_239 == var_246);
        if (var_247) {
            // c = PENT_COS_2                                                                 <L 81>
            var_248 = wp::copy(var_6);
            // s = PENT_SIN_2                                                                 <L 82>
            var_249 = wp::copy(var_7);
        }
        if (!var_247) {
            // elif e == 3:                                                                   <L 83>
            var_251 = (var_239 == var_250);
            if (var_251) {
                // c = PENT_COS_3                                                             <L 84>
                var_252 = wp::copy(var_8);
                // s = PENT_SIN_3                                                             <L 85>
                var_253 = wp::copy(var_9);
            }
            if (!var_251) {
                // elif e == 4:                                                               <L 86>
                var_255 = (var_239 == var_254);
                if (var_255) {
                    // c = PENT_COS_4                                                         <L 87>
                    var_256 = wp::copy(var_10);
                    // s = PENT_SIN_4                                                         <L 88>
                    var_257 = wp::copy(var_11);
                }
                var_258 = wp::where(var_255, var_256, var_240);
                var_259 = wp::where(var_255, var_257, var_241);
            }
            var_260 = wp::where(var_251, var_252, var_258);
            var_261 = wp::where(var_251, var_253, var_259);
        }
        var_262 = wp::where(var_247, var_248, var_260);
        var_263 = wp::where(var_247, var_249, var_261);
    }
    var_264 = wp::where(var_243, var_244, var_262);
    var_265 = wp::where(var_243, var_245, var_263);
    // cos_tilt = COS_TILT_ANGLE                                                              <L 90>
    var_266 = wp::copy(var_71);
    // c_sin = c * SIN_TILT_ANGLE                                                             <L 91>
    var_267 = wp::mul(var_264, var_73);
    // s_sin = s * SIN_TILT_ANGLE                                                             <L 92>
    var_268 = wp::mul(var_265, var_73);
    // dir_a = normal * cos_tilt + c_sin * tangent_a + s_sin * tangent_b                      <L 95>
    var_269 = wp::mul(var_normal, var_266);
    var_270 = wp::mul(var_267, var_16);
    var_271 = wp::add(var_269, var_270);
    var_272 = wp::mul(var_268, var_17);
    var_273 = wp::add(var_271, var_272);
    // pt_a_3d = support_func(geom_a, dir_a, data_provider)                                   <L 96>
    var_274 = support_map_lean_0(var_geom_a, var_273, var_data_provider);
    // projected_a = pt_a_3d - center                                                         <L 97>
    var_275 = wp::sub(var_274, var_22);
    // pt_a_2d = wp.vec2(wp.dot(tangent_a, projected_a), wp.dot(tangent_b, projected_a))       <L 98>
    var_276 = wp::dot(var_16, var_275);
    var_277 = wp::dot(var_17, var_275);
    var_278 = wp::vec_t<2, wp::float32>(var_276, var_277);
    // a_count, was_added_a = add_avoid_duplicates_vec2(a_buffer, a_count, pt_a_2d, EPS)       <L 99>
    add_avoid_duplicates_vec2_0(var_36, var_215, var_278, var_86, var_279, var_280);
    // if was_added_a:                                                                        <L 100>
    if (var_280) {
        // plane_tracker_a = update_incremental_plane_tracker(plane_tracker_a, pt_a_3d, a_count - 1)       <L 101>
        var_282 = wp::sub(var_279, var_281);
        var_283 = update_incremental_plane_tracker_0(var_220, var_274, var_282);
    }
    var_284 = wp::where(var_280, var_283, var_220);
    // local_dir_b = local_normal_b * cos_tilt + c_sin * local_ta_b + s_sin * local_tb_b       <L 104>
    var_285 = wp::mul(var_39, var_266);
    var_286 = wp::mul(var_267, var_41);
    var_287 = wp::add(var_285, var_286);
    var_288 = wp::mul(var_268, var_43);
    var_289 = wp::add(var_287, var_288);
    // pt_b_local = support_func(geom_b, local_dir_b, data_provider)                          <L 105>
    var_290 = support_map_lean_0(var_geom_b, var_289, var_data_provider);
    // pt_b_3d = wp.quat_rotate(relative_orientation_b, pt_b_local) + relative_position_b       <L 106>
    var_291 = wp::quat_rotate(var_relative_orientation_b, var_290);
    var_292 = wp::add(var_291, var_relative_position_b);
    // projected_b = pt_b_3d - center                                                         <L 107>
    var_293 = wp::sub(var_292, var_22);
    // pt_b_2d = wp.vec2(wp.dot(tangent_a, projected_b), wp.dot(tangent_b, projected_b))       <L 108>
    var_294 = wp::dot(var_16, var_293);
    var_295 = wp::dot(var_17, var_293);
    var_296 = wp::vec_t<2, wp::float32>(var_294, var_295);
    // b_count, was_added_b = add_avoid_duplicates_vec2(b_buffer, b_count, pt_b_2d, EPS)       <L 109>
    add_avoid_duplicates_vec2_0(var_25, var_233, var_296, var_86, var_297, var_298);
    // if was_added_b:                                                                        <L 110>
    if (var_298) {
        // plane_tracker_b = update_incremental_plane_tracker(plane_tracker_b, pt_b_3d, b_count - 1)       <L 111>
        var_300 = wp::sub(var_297, var_299);
        var_301 = update_incremental_plane_tracker_0(var_238, var_292, var_300);
    }
    var_302 = wp::where(var_298, var_301, var_238);
    // c = PENT_COS_0                                                                         <L 75>
    var_304 = wp::copy(var_1);
    // s = PENT_SIN_0                                                                         <L 76>
    var_305 = wp::copy(var_3);
    // if e == 1:                                                                             <L 77>
    var_307 = (var_303 == var_306);
    if (var_307) {
        // c = PENT_COS_1                                                                     <L 78>
        var_308 = wp::copy(var_4);
        // s = PENT_SIN_1                                                                     <L 79>
        var_309 = wp::copy(var_5);
    }
    if (!var_307) {
        // elif e == 2:                                                                       <L 80>
        var_311 = (var_303 == var_310);
        if (var_311) {
            // c = PENT_COS_2                                                                 <L 81>
            var_312 = wp::copy(var_6);
            // s = PENT_SIN_2                                                                 <L 82>
            var_313 = wp::copy(var_7);
        }
        if (!var_311) {
            // elif e == 3:                                                                   <L 83>
            var_315 = (var_303 == var_314);
            if (var_315) {
                // c = PENT_COS_3                                                             <L 84>
                var_316 = wp::copy(var_8);
                // s = PENT_SIN_3                                                             <L 85>
                var_317 = wp::copy(var_9);
            }
            if (!var_315) {
                // elif e == 4:                                                               <L 86>
                var_319 = (var_303 == var_318);
                if (var_319) {
                    // c = PENT_COS_4                                                         <L 87>
                    var_320 = wp::copy(var_10);
                    // s = PENT_SIN_4                                                         <L 88>
                    var_321 = wp::copy(var_11);
                }
                var_322 = wp::where(var_319, var_320, var_304);
                var_323 = wp::where(var_319, var_321, var_305);
            }
            var_324 = wp::where(var_315, var_316, var_322);
            var_325 = wp::where(var_315, var_317, var_323);
        }
        var_326 = wp::where(var_311, var_312, var_324);
        var_327 = wp::where(var_311, var_313, var_325);
    }
    var_328 = wp::where(var_307, var_308, var_326);
    var_329 = wp::where(var_307, var_309, var_327);
    // cos_tilt = COS_TILT_ANGLE                                                              <L 90>
    var_330 = wp::copy(var_71);
    // c_sin = c * SIN_TILT_ANGLE                                                             <L 91>
    var_331 = wp::mul(var_328, var_73);
    // s_sin = s * SIN_TILT_ANGLE                                                             <L 92>
    var_332 = wp::mul(var_329, var_73);
    // dir_a = normal * cos_tilt + c_sin * tangent_a + s_sin * tangent_b                      <L 95>
    var_333 = wp::mul(var_normal, var_330);
    var_334 = wp::mul(var_331, var_16);
    var_335 = wp::add(var_333, var_334);
    var_336 = wp::mul(var_332, var_17);
    var_337 = wp::add(var_335, var_336);
    // pt_a_3d = support_func(geom_a, dir_a, data_provider)                                   <L 96>
    var_338 = support_map_lean_0(var_geom_a, var_337, var_data_provider);
    // projected_a = pt_a_3d - center                                                         <L 97>
    var_339 = wp::sub(var_338, var_22);
    // pt_a_2d = wp.vec2(wp.dot(tangent_a, projected_a), wp.dot(tangent_b, projected_a))       <L 98>
    var_340 = wp::dot(var_16, var_339);
    var_341 = wp::dot(var_17, var_339);
    var_342 = wp::vec_t<2, wp::float32>(var_340, var_341);
    // a_count, was_added_a = add_avoid_duplicates_vec2(a_buffer, a_count, pt_a_2d, EPS)       <L 99>
    add_avoid_duplicates_vec2_0(var_36, var_279, var_342, var_86, var_343, var_344);
    // if was_added_a:                                                                        <L 100>
    if (var_344) {
        // plane_tracker_a = update_incremental_plane_tracker(plane_tracker_a, pt_a_3d, a_count - 1)       <L 101>
        var_346 = wp::sub(var_343, var_345);
        var_347 = update_incremental_plane_tracker_0(var_284, var_338, var_346);
    }
    var_348 = wp::where(var_344, var_347, var_284);
    // local_dir_b = local_normal_b * cos_tilt + c_sin * local_ta_b + s_sin * local_tb_b       <L 104>
    var_349 = wp::mul(var_39, var_330);
    var_350 = wp::mul(var_331, var_41);
    var_351 = wp::add(var_349, var_350);
    var_352 = wp::mul(var_332, var_43);
    var_353 = wp::add(var_351, var_352);
    // pt_b_local = support_func(geom_b, local_dir_b, data_provider)                          <L 105>
    var_354 = support_map_lean_0(var_geom_b, var_353, var_data_provider);
    // pt_b_3d = wp.quat_rotate(relative_orientation_b, pt_b_local) + relative_position_b       <L 106>
    var_355 = wp::quat_rotate(var_relative_orientation_b, var_354);
    var_356 = wp::add(var_355, var_relative_position_b);
    // projected_b = pt_b_3d - center                                                         <L 107>
    var_357 = wp::sub(var_356, var_22);
    // pt_b_2d = wp.vec2(wp.dot(tangent_a, projected_b), wp.dot(tangent_b, projected_b))       <L 108>
    var_358 = wp::dot(var_16, var_357);
    var_359 = wp::dot(var_17, var_357);
    var_360 = wp::vec_t<2, wp::float32>(var_358, var_359);
    // b_count, was_added_b = add_avoid_duplicates_vec2(b_buffer, b_count, pt_b_2d, EPS)       <L 109>
    add_avoid_duplicates_vec2_0(var_25, var_297, var_360, var_86, var_361, var_362);
    // if was_added_b:                                                                        <L 110>
    if (var_362) {
        // plane_tracker_b = update_incremental_plane_tracker(plane_tracker_b, pt_b_3d, b_count - 1)       <L 111>
        var_364 = wp::sub(var_361, var_363);
        var_365 = update_incremental_plane_tracker_0(var_302, var_356, var_364);
    }
    var_366 = wp::where(var_362, var_365, var_302);
    // normal_world = wp.quat_rotate(orientation_a, normal)                                   <L 114>
    var_367 = wp::quat_rotate(var_orientation_a, var_normal);
    // position_a_ws = position_a_world                                                       <L 117>
    var_368 = wp::copy(var_position_a_world);
    // position_b_ws = wp.quat_rotate(orientation_a, relative_position_b) + position_a_world       <L 118>
    var_369 = wp::quat_rotate(var_orientation_a, var_relative_position_b);
    var_370 = wp::add(var_369, var_position_a_world);
    // quaternion_a_ws = orientation_a                                                        <L 119>
    var_371 = wp::copy(var_orientation_a);
    // quaternion_b_ws = orientation_a * relative_orientation_b                               <L 120>
    var_372 = wp::mul(var_orientation_a, var_relative_orientation_b);
    // if a_count < 2 or b_count < 2:                                                         <L 122>
    var_375 = (var_343 < var_374);
    var_373 = var_375;
    if (!var_373) {
        var_377 = (var_361 < var_376);
        var_373 = var_373 || var_377;
    }
    if (var_373) {
        // count_out = 0                                                                      <L 123>
        // normal_dot = 0.0                                                                   <L 124>
    }
    if (!var_373) {
        // projector_a, projector_b = create_body_projectors(plane_tracker_a, p_a, plane_tracker_b, p_b, normal)       <L 126>
        create_body_projectors_0(var_348, var_p_a, var_366, var_p_b, var_normal, var_380, var_381);
        // if excess_normal_deviation(normal, projector_a.normal) or excess_normal_deviation(       <L 128>
        var_383 = &((var_380).normal);
        var_385 = wp::load(var_383);
        var_384 = excess_normal_deviation_0(var_normal, var_385);
        var_382 = var_384;
        if (!var_382) {
            // normal, projector_b.normal                                                     <L 129>
            var_386 = &((var_381).normal);
            var_388 = wp::load(var_386);
            var_387 = excess_normal_deviation_0(var_normal, var_388);
            var_382 = var_382 || var_387;
        }
        if (var_382) {
            // count_out = 0                                                                  <L 131>
            // normal_dot = 0.0                                                               <L 132>
        }
        if (!var_382) {
            // num_manifold_points, normal_dot = extract_4_point_contact_manifolds(           <L 134>
            // a_buffer,                                                                      <L 135>
            // a_count,                                                                       <L 136>
            // b_buffer,                                                                      <L 137>
            // b_count,                                                                       <L 138>
            // normal,                                                                        <L 139>
            // tangent_a,                                                                     <L 140>
            // tangent_b,                                                                     <L 141>
            // center,                                                                        <L 142>
            // projector_a,                                                                   <L 143>
            // projector_b,                                                                   <L 144>
            // orientation_a,                                                                 <L 145>
            // position_a_world,                                                              <L 146>
            // normal_world,                                                                  <L 147>
            // writer_data,                                                                   <L 148>
            // contact_template,                                                              <L 149>
            // geom_a,                                                                        <L 150>
            // geom_b,                                                                        <L 151>
            // position_a_ws,                                                                 <L 152>
            // position_b_ws,                                                                 <L 153>
            // quaternion_a_ws,                                                               <L 154>
            // quaternion_b_ws,                                                               <L 155>
            create_build_manifold__locals__extract_4_point_contact_manifolds_5(var_36, var_343, var_25, var_361, var_normal, var_16, var_17, var_22, var_380, var_381, var_orientation_a, var_position_a_world, var_367, var_writer_data, var_contact_template, var_geom_a, var_geom_b, var_368, var_370, var_371, var_372, var_391, var_392);
            // count_out = wp.min(num_manifold_points, 4)                                     <L 157>
            var_394 = wp::min(var_391, var_393);
        }
        var_395 = wp::where(var_382, var_389, var_394);
        var_396 = wp::where(var_382, var_390, var_392);
    }
    var_397 = wp::where(var_373, var_378, var_395);
    var_398 = wp::where(var_373, var_379, var_396);
    // if should_include_deepest_contact(normal_dot) or count_out == 0:                       <L 159>
    var_400 = should_include_deepest_contact_0(var_398);
    var_399 = var_400;
    if (!var_399) {
        var_402 = (var_397 == var_401);
        var_399 = var_399 || var_402;
    }
    if (var_399) {
        // deepest_center_local = 0.5 * (p_a + p_b)                                           <L 160>
        var_404 = wp::add(var_p_a, var_p_b);
        var_405 = wp::mul(var_403, var_404);
        // deepest_signed_distance = wp.dot(p_b - p_a, normal)                                <L 161>
        var_406 = wp::sub(var_p_b, var_p_a);
        var_407 = wp::dot(var_406, var_normal);
        // deepest_center_world = wp.quat_rotate(orientation_a, deepest_center_local) + position_a_world       <L 163>
        var_408 = wp::quat_rotate(var_orientation_a, var_405);
        var_409 = wp::add(var_408, var_position_a_world);
        // contact_data = contact_template                                                    <L 165>
        var_410 = wp::copy(var_contact_template);
        // contact_data.contact_point_center = deepest_center_world                           <L 166>
        var_410.contact_point_center = var_409;
        // contact_data.contact_normal_a_to_b = normal_world                                  <L 167>
        var_410.contact_normal_a_to_b = var_367;
        // contact_data.contact_distance = deepest_signed_distance                            <L 168>
        var_410.contact_distance = var_407;
        // contact_data.sort_sub_key = (contact_template.sort_sub_key << 3) | count_out       <L 169>
        var_411 = &((var_contact_template).sort_sub_key);
        var_414 = wp::load(var_411);
        var_413 = wp::lshift(var_414, var_412);
        var_415 = wp::bit_or(var_413, var_397);
        var_410.sort_sub_key = var_415;
        // contact_data = post_process_contact(                                               <L 171>
        // contact_data, geom_a, position_a_ws, quaternion_a_ws, geom_b, position_b_ws, quaternion_b_ws       <L 172>
        var_416 = post_process_minkowski_only_0(var_410, var_geom_a, var_368, var_371, var_geom_b, var_370, var_372);
        // writer_func(contact_data, writer_data, -1)                                         <L 174>
        write_contact_0(var_416, var_writer_data, var_417);
        // count_out += 1                                                                     <L 176>
        var_419 = wp::add(var_397, var_418);
    }
    var_420 = wp::where(var_399, var_419, var_397);
    // return count_out                                                                       <L 178>
    return var_420;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_convex.py:0
static CUDA_CALLABLE wp::int32 create_solve_convex_multi_contact__locals__solve_convex_multi_contact_4(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_a,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_a,
    wp::vec_t<3, wp::float32> var_position_b,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    wp::float32 var_contact_threshold,
    bool var_skip_multi_contact,
    ContactWriterData_099b2624 var_writer_data,
    ContactData_40360d7c var_contact_template)
{
    //---------
    // primal vars
    wp::quat_t<wp::float32> var_0;
    wp::quat_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32* var_4;
    wp::float32* var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0001;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    wp::float32 var_12;
    bool var_13;
    const wp::float32 var_14 = 2.0;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.0;
    wp::float32 var_17;
    wp::float32 var_18;
    bool var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 30;
    const wp::float32 var_25 = 1e-05;
    const bool var_26 = false;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::float32 var_29 = 0.5;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::float32 var_35 = 0.0;
    bool var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::float32 var_40;
    const wp::int32 var_41 = 30;
    const wp::float32 var_42 = 0.0001;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::float32 var_46;
    bool var_47;
    bool var_48;
    const wp::float32 var_49 = 0.5;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    ContactData_40360d7c var_55;
    wp::int32* var_56;
    const wp::int32 var_57 = 3;
    wp::int32 var_58;
    wp::int32 var_59;
    ContactData_40360d7c var_60;
    const wp::int32 var_61 = -1;
    const wp::int32 var_62 = 1;
    wp::int32 var_63;
    //---------
    // forward
    // def solve_convex_multi_contact(                                                        <L 1>
    // relative_orientation_b = wp.quat_inverse(orientation_a) * orientation_b                <L 15>
    var_0 = wp::quat_inverse(var_orientation_a);
    var_1 = wp::mul(var_0, var_orientation_b);
    // relative_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)       <L 16>
    var_2 = wp::sub(var_position_b, var_position_a);
    var_3 = wp::quat_rotate_inv(var_orientation_a, var_2);
    // margin_sum = contact_template.margin_a + contact_template.margin_b                     <L 24>
    var_4 = &((var_contact_template).margin_a);
    var_5 = &((var_contact_template).margin_b);
    var_7 = wp::load(var_4);
    var_8 = wp::load(var_5);
    var_6 = wp::add(var_7, var_8);
    // eps = 1.0e-4                                                                           <L 25>
    // if margin_sum <= 0.0:                                                                  <L 26>
    var_11 = (var_6 <= var_10);
    if (var_11) {
        // enlarge = eps                                                                      <L 27>
        var_12 = wp::copy(var_9);
    }
    if (!var_11) {
        // elif margin_sum < eps:                                                             <L 28>
        var_13 = (var_6 < var_9);
        if (var_13) {
            // enlarge = 2.0 * eps                                                            <L 29>
            var_15 = wp::mul(var_14, var_9);
        }
        if (!var_13) {
            // enlarge = 0.0                                                                  <L 31>
        }
        var_17 = wp::where(var_13, var_15, var_16);
    }
    var_18 = wp::where(var_11, var_12, var_17);
    // collision, point_a, point_b, normal, penetration = wp.static(solve_mpr.core)(          <L 35>
    // geom_a,                                                                                <L 36>
    // geom_b,                                                                                <L 37>
    // relative_orientation_b,                                                                <L 38>
    // relative_position_b,                                                                   <L 39>
    // enlarge,                                                                               <L 40>
    // data_provider,                                                                         <L 41>
    create_solve_mpr__locals__solve_mpr_core_8(var_geom_a, var_geom_b, var_1, var_3, var_18, var_data_provider, var_24, var_25, var_19, var_20, var_21, var_22, var_23);
    // if collision:                                                                          <L 44>
    if (var_19) {
        // if wp.static(has_penetration_refiner):                                             <L 45>
        // signed_distance = -penetration + enlarge                                           <L 58>
        var_27 = wp::neg(var_23);
        var_28 = wp::add(var_27, var_18);
        // half_enlarge = enlarge * 0.5                                                       <L 62>
        var_30 = wp::mul(var_18, var_29);
        // point_a = point_a - normal * half_enlarge                                          <L 63>
        var_31 = wp::mul(var_22, var_30);
        var_32 = wp::sub(var_20, var_31);
        // point_b = point_b + normal * half_enlarge                                          <L 64>
        var_33 = wp::mul(var_22, var_30);
        var_34 = wp::add(var_21, var_33);
    }
    if (!var_19) {
        // _separated, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(       <L 67>
        // geom_a,                                                                            <L 68>
        // geom_b,                                                                            <L 69>
        // relative_orientation_b,                                                            <L 70>
        // relative_position_b,                                                               <L 71>
        // 0.0,                                                                               <L 72>
        // data_provider,                                                                     <L 73>
        create_solve_closest_distance__locals__solve_closest_distance_core_8(var_geom_a, var_geom_b, var_1, var_3, var_35, var_data_provider, var_41, var_42, var_36, var_37, var_38, var_39, var_40);
    }
    var_43 = wp::where(var_19, var_32, var_37);
    var_44 = wp::where(var_19, var_34, var_38);
    var_45 = wp::where(var_19, var_22, var_39);
    var_46 = wp::where(var_19, var_28, var_40);
    // if skip_multi_contact or signed_distance > contact_threshold:                          <L 76>
    var_47 = var_skip_multi_contact;
    if (!var_47) {
        var_48 = (var_46 > var_contact_threshold);
        var_47 = var_47 || var_48;
    }
    if (var_47) {
        // point = 0.5 * (point_a + point_b)                                                  <L 78>
        var_50 = wp::add(var_43, var_44);
        var_51 = wp::mul(var_49, var_50);
        // point = wp.quat_rotate(orientation_a, point) + position_a                          <L 79>
        var_52 = wp::quat_rotate(var_orientation_a, var_51);
        var_53 = wp::add(var_52, var_position_a);
        // normal_ws = wp.quat_rotate(orientation_a, normal)                                  <L 80>
        var_54 = wp::quat_rotate(var_orientation_a, var_45);
        // contact_data = contact_template                                                    <L 82>
        var_55 = wp::copy(var_contact_template);
        // contact_data.contact_point_center = point                                          <L 83>
        var_55.contact_point_center = var_53;
        // contact_data.contact_normal_a_to_b = normal_ws                                     <L 84>
        var_55.contact_normal_a_to_b = var_54;
        // contact_data.contact_distance = signed_distance                                    <L 85>
        var_55.contact_distance = var_46;
        // contact_data.sort_sub_key = contact_template.sort_sub_key << 3                     <L 86>
        var_56 = &((var_contact_template).sort_sub_key);
        var_59 = wp::load(var_56);
        var_58 = wp::lshift(var_59, var_57);
        var_55.sort_sub_key = var_58;
        // contact_data = post_process_contact(                                               <L 87>
        // contact_data, geom_a, position_a, orientation_a, geom_b, position_b, orientation_b       <L 88>
        var_60 = post_process_minkowski_only_0(var_55, var_geom_a, var_position_a, var_orientation_a, var_geom_b, var_position_b, var_orientation_b);
        // writer_func(contact_data, writer_data, -1)                                         <L 90>
        write_contact_0(var_60, var_writer_data, var_61);
        // return 1                                                                           <L 91>
        return var_62;
    }
    // count = wp.static(                                                                     <L 95>
    // geom_a,                                                                                <L 98>
    // geom_b,                                                                                <L 99>
    // orientation_a,                                                                         <L 100>
    // position_a,                                                                            <L 101>
    // relative_orientation_b,                                                                <L 102>
    // relative_position_b,                                                                   <L 103>
    // point_a,                                                                               <L 104>
    // point_b,                                                                               <L 105>
    // normal,                                                                                <L 106>
    // data_provider,                                                                         <L 107>
    // writer_data,                                                                           <L 108>
    // contact_template,                                                                      <L 109>
    var_63 = create_build_manifold__locals__build_manifold_5(var_geom_a, var_geom_b, var_orientation_a, var_position_a, var_1, var_3, var_43, var_44, var_45, var_data_provider, var_writer_data, var_contact_template);
    // return count                                                                           <L 112>
    return var_63;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:0
static CUDA_CALLABLE void create_compute_gjk_mpr_contacts__locals__compute_gjk_mpr_contacts_3(
    GenericShapeData_ba895271 var_shape_a_data,
    GenericShapeData_ba895271 var_shape_b_data,
    wp::quat_t<wp::float32> var_rot_a,
    wp::quat_t<wp::float32> var_rot_b,
    wp::vec_t<3, wp::float32> var_pos_a_adjusted,
    wp::vec_t<3, wp::float32> var_pos_b_adjusted,
    wp::float32 var_rigid_gap,
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::float32 var_margin_a,
    wp::float32 var_margin_b,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    ContactWriterData_099b2624 var_writer_data,
    wp::int32 var_sort_sub_key)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.0001;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    bool var_11;
    const wp::int32 var_12 = 3;
    bool var_13;
    const wp::int32 var_14 = 4;
    bool var_15;
    wp::vec_t<3, wp::float32>* var_16;
    const wp::int32 var_17 = 0;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32>* var_20;
    const wp::int32 var_21 = 0;
    wp::float32* var_22;
    wp::float32 var_23;
    bool var_24;
    const wp::int32 var_25 = 3;
    bool var_26;
    const wp::int32 var_27 = 4;
    bool var_28;
    wp::vec_t<3, wp::float32>* var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32>* var_33;
    const wp::int32 var_34 = 0;
    wp::float32* var_35;
    wp::float32 var_36;
    ContactData_40360d7c var_37;
    const bool var_38 = true;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    bool var_43;
    const wp::int32 var_44 = 3;
    bool var_45;
    const wp::int32 var_46 = 3;
    bool var_47;
    const wp::int32 var_48 = 5;
    bool var_49;
    const wp::int32 var_50 = 5;
    bool var_51;
    wp::int32 var_52;
    //---------
    // forward
    // def compute_gjk_mpr_contacts(                                                          <L 1>
    // radius_eff_a = float(0.0)                                                              <L 38>
    var_1 = wp::float(var_0);
    // radius_eff_b = float(0.0)                                                              <L 39>
    var_3 = wp::float(var_2);
    // small_radius = 0.0001                                                                  <L 41>
    // type_a = shape_a_data.shape_type                                                       <L 44>
    var_5 = &((var_shape_a_data).shape_type);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // type_b = shape_b_data.shape_type                                                       <L 45>
    var_8 = &((var_shape_b_data).shape_type);
    var_10 = wp::load(var_8);
    var_9 = wp::copy(var_10);
    // if type_a == GeoType.SPHERE or type_a == GeoType.CAPSULE:                              <L 48>
    var_13 = (var_6 == var_12);
    var_11 = var_13;
    if (!var_11) {
        var_15 = (var_6 == var_14);
        var_11 = var_11 || var_15;
    }
    if (var_11) {
        // radius_eff_a = shape_a_data.scale[0]                                               <L 49>
        var_16 = &((var_shape_a_data).scale);
        var_19 = wp::load(var_16);
        var_18 = wp::extract(var_19, var_17);
        // shape_a_data.scale[0] = small_radius                                               <L 50>
        var_20 = &((var_shape_a_data).scale);
        var_22 = wp::indexref(var_20, var_21);
        wp::store(var_22, var_4);
    }
    var_23 = wp::where(var_11, var_18, var_1);
    // if type_b == GeoType.SPHERE or type_b == GeoType.CAPSULE:                              <L 52>
    var_26 = (var_9 == var_25);
    var_24 = var_26;
    if (!var_24) {
        var_28 = (var_9 == var_27);
        var_24 = var_24 || var_28;
    }
    if (var_24) {
        // radius_eff_b = shape_b_data.scale[0]                                               <L 53>
        var_29 = &((var_shape_b_data).scale);
        var_32 = wp::load(var_29);
        var_31 = wp::extract(var_32, var_30);
        // shape_b_data.scale[0] = small_radius                                               <L 54>
        var_33 = &((var_shape_b_data).scale);
        var_35 = wp::indexref(var_33, var_34);
        wp::store(var_35, var_4);
    }
    var_36 = wp::where(var_24, var_31, var_3);
    // contact_template = ContactData()                                                       <L 57>
    var_37 = ContactData_40360d7c();
    // contact_template.radius_eff_a = radius_eff_a                                           <L 58>
    var_37.radius_eff_a = var_23;
    // contact_template.radius_eff_b = radius_eff_b                                           <L 59>
    var_37.radius_eff_b = var_36;
    // contact_template.margin_a = margin_a                                                   <L 60>
    var_37.margin_a = var_margin_a;
    // contact_template.margin_b = margin_b                                                   <L 61>
    var_37.margin_b = var_margin_b;
    // contact_template.shape_a = shape_a                                                     <L 62>
    var_37.shape_a = var_shape_a;
    // contact_template.shape_b = shape_b                                                     <L 63>
    var_37.shape_b = var_shape_b;
    // contact_template.gap_sum = rigid_gap                                                   <L 64>
    var_37.gap_sum = var_rigid_gap;
    // contact_template.sort_sub_key = sort_sub_key                                           <L 65>
    var_37.sort_sub_key = var_sort_sub_key;
    // if wp.static(ENABLE_MULTI_CONTACT):                                                    <L 67>
    // wp.static(                                                                             <L 68>
    // shape_a_data,                                                                          <L 77>
    // shape_b_data,                                                                          <L 78>
    // rot_a,                                                                                 <L 79>
    // rot_b,                                                                                 <L 80>
    // pos_a_adjusted,                                                                        <L 81>
    // pos_b_adjusted,                                                                        <L 82>
    // data_provider,                                                                         <L 83>
    // rigid_gap + radius_eff_a + radius_eff_b + margin_a + margin_b,                         <L 84>
    var_39 = wp::add(var_rigid_gap, var_23);
    var_40 = wp::add(var_39, var_36);
    var_41 = wp::add(var_40, var_margin_a);
    var_42 = wp::add(var_41, var_margin_b);
    // type_a == GeoType.SPHERE                                                               <L 85>
    var_45 = (var_6 == var_44);
    var_43 = var_45;
    if (!var_43) {
        // or type_b == GeoType.SPHERE                                                        <L 86>
        var_47 = (var_9 == var_46);
        var_43 = var_43 || var_47;
    }
    if (!var_43) {
        // or type_a == GeoType.ELLIPSOID                                                     <L 87>
        var_49 = (var_6 == var_48);
        var_43 = var_43 || var_49;
    }
    if (!var_43) {
        // or type_b == GeoType.ELLIPSOID,                                                    <L 88>
        var_51 = (var_9 == var_50);
        var_43 = var_43 || var_51;
    }
    // writer_data,                                                                           <L 89>
    // contact_template,                                                                      <L 90>
    var_52 = create_solve_convex_multi_contact__locals__solve_convex_multi_contact_4(var_shape_a_data, var_shape_b_data, var_rot_a, var_rot_b, var_pos_a_adjusted, var_pos_b_adjusted, var_data_provider, var_42, var_43, var_writer_data, var_37);
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:0
static CUDA_CALLABLE void create_find_contacts__locals__find_contacts_1(
    wp::vec_t<3, wp::float32> var_pos_a,
    wp::vec_t<3, wp::float32> var_pos_b,
    wp::quat_t<wp::float32> var_quat_a,
    wp::quat_t<wp::float32> var_quat_b,
    GenericShapeData_ba895271 var_shape_data_a,
    GenericShapeData_ba895271 var_shape_data_b,
    bool var_is_infinite_plane_a,
    bool var_is_infinite_plane_b,
    wp::float32 var_bsphere_radius_a,
    wp::float32 var_bsphere_radius_b,
    wp::float32 var_rigid_gap,
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::float32 var_margin_a,
    wp::float32 var_margin_b,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    ContactWriterData_099b2624 var_writer_data)
{
    //---------
    // primal vars
    wp::array_t<wp::int32>* var_0;
    const wp::int32 var_1 = 0;
    wp::int32* var_2;
    wp::array_t<wp::int32> var_3;
    wp::int32* var_4;
    bool var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    GenericShapeData_ba895271 var_10;
    wp::vec_t<3, wp::float32> var_11;
    GenericShapeData_ba895271 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::float32 var_15;
    GenericShapeData_ba895271 var_16;
    wp::vec_t<3, wp::float32> var_17;
    GenericShapeData_ba895271 var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 0;
    //---------
    // forward
    // def find_contacts(                                                                     <L 1>
    // if writer_data.contact_count[0] >= writer_data.contact_max:                            <L 44>
    var_0 = &((var_writer_data).contact_count);
    var_3 = wp::load(var_0);
    var_2 = wp::address(var_3, var_1);
    var_4 = &((var_writer_data).contact_max);
    var_6 = wp::load(var_2);
    var_7 = wp::load(var_4);
    var_5 = (var_6 >= var_7);
    if (var_5) {
        // return                                                                             <L 45>
        return;
    }
    // pos_a_adjusted = pos_a                                                                 <L 50>
    var_8 = wp::copy(var_pos_a);
    // if is_infinite_plane_a:                                                                <L 51>
    if (var_is_infinite_plane_a) {
        // shape_data_a, pos_a_adjusted = convert_infinite_plane_to_cube(                     <L 54>
        // shape_data_a, quat_a, pos_a, pos_b, bsphere_radius_b + rigid_gap                   <L 55>
        var_9 = wp::add(var_bsphere_radius_b, var_rigid_gap);
        convert_infinite_plane_to_cube_0(var_shape_data_a, var_quat_a, var_pos_a, var_pos_b, var_9, var_10, var_11);
    }
    var_12 = wp::where(var_is_infinite_plane_a, var_10, var_shape_data_a);
    var_13 = wp::where(var_is_infinite_plane_a, var_11, var_8);
    // pos_b_adjusted = pos_b                                                                 <L 58>
    var_14 = wp::copy(var_pos_b);
    // if is_infinite_plane_b:                                                                <L 59>
    if (var_is_infinite_plane_b) {
        // shape_data_b, pos_b_adjusted = convert_infinite_plane_to_cube(                     <L 62>
        // shape_data_b, quat_b, pos_b, pos_a, bsphere_radius_a + rigid_gap                   <L 63>
        var_15 = wp::add(var_bsphere_radius_a, var_rigid_gap);
        convert_infinite_plane_to_cube_0(var_shape_data_b, var_quat_b, var_pos_b, var_pos_a, var_15, var_16, var_17);
    }
    var_18 = wp::where(var_is_infinite_plane_b, var_16, var_shape_data_b);
    var_19 = wp::where(var_is_infinite_plane_b, var_17, var_14);
    // wp.static(                                                                             <L 67>
    // shape_data_a,                                                                          <L 75>
    // shape_data_b,                                                                          <L 76>
    // quat_a,                                                                                <L 77>
    // quat_b,                                                                                <L 78>
    // pos_a_adjusted,                                                                        <L 79>
    // pos_b_adjusted,                                                                        <L 80>
    // rigid_gap,                                                                             <L 81>
    // shape_a,                                                                               <L 82>
    // shape_b,                                                                               <L 83>
    // margin_a,                                                                              <L 84>
    // margin_b,                                                                              <L 85>
    // data_provider,                                                                         <L 86>
    // writer_data,                                                                           <L 87>
    create_compute_gjk_mpr_contacts__locals__compute_gjk_mpr_contacts_3(var_12, var_18, var_quat_a, var_quat_b, var_13, var_19, var_rigid_gap, var_shape_a, var_shape_b, var_margin_a, var_margin_b, var_data_provider, var_writer_data, var_20);
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:81
static CUDA_CALLABLE void adj_pack_mesh_ptr_0(
    wp::uint64 var_ptr,
    wp::uint64 & adj_ptr,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:768
static CUDA_CALLABLE void adj_extract_shape_data_0(
    wp::int32 var_shape_idx,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::int32> var_shape_types,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::uint64> var_shape_source,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::quat_t<wp::float32> & ret_1,
    GenericShapeData_ba895271 & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::float32 & ret_4,
    wp::int32 & adj_shape_idx,
    wp::array_t<wp::transform_t<wp::float32>> & adj_shape_transform,
    wp::array_t<wp::int32> & adj_shape_types,
    wp::array_t<wp::vec_t<4, wp::float32>> & adj_shape_data,
    wp::array_t<wp::uint64> & adj_shape_source,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::quat_t<wp::float32> & adj_ret_1,
    GenericShapeData_ba895271 & adj_ret_2,
    wp::vec_t<3, wp::float32> & adj_ret_3,
    wp::float32 & adj_ret_4)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:516
static CUDA_CALLABLE void adj_compute_bounding_sphere_from_aabb_0(
    wp::vec_t<3, wp::float32> var_aabb_lower,
    wp::vec_t<3, wp::float32> var_aabb_upper,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_aabb_lower,
    wp::vec_t<3, wp::float32> & adj_aabb_upper,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:594
static CUDA_CALLABLE void adj_check_infinite_plane_bsphere_overlap_0(
    GenericShapeData_ba895271 var_shape_data_a,
    GenericShapeData_ba895271 var_shape_data_b,
    wp::vec_t<3, wp::float32> var_pos_a,
    wp::vec_t<3, wp::float32> var_pos_b,
    wp::quat_t<wp::float32> var_quat_a,
    wp::quat_t<wp::float32> var_quat_b,
    wp::vec_t<3, wp::float32> var_bsphere_center_a,
    wp::vec_t<3, wp::float32> var_bsphere_center_b,
    wp::float32 var_bsphere_radius_a,
    wp::float32 var_bsphere_radius_b,
    GenericShapeData_ba895271 & adj_shape_data_a,
    GenericShapeData_ba895271 & adj_shape_data_b,
    wp::vec_t<3, wp::float32> & adj_pos_a,
    wp::vec_t<3, wp::float32> & adj_pos_b,
    wp::quat_t<wp::float32> & adj_quat_a,
    wp::quat_t<wp::float32> & adj_quat_b,
    wp::vec_t<3, wp::float32> & adj_bsphere_center_a,
    wp::vec_t<3, wp::float32> & adj_bsphere_center_b,
    wp::float32 & adj_bsphere_radius_a,
    wp::float32 & adj_bsphere_radius_b,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:530
static CUDA_CALLABLE void adj_convert_infinite_plane_to_cube_0(
    GenericShapeData_ba895271 var_shape_data,
    wp::quat_t<wp::float32> var_plane_rotation,
    wp::vec_t<3, wp::float32> var_plane_position,
    wp::vec_t<3, wp::float32> var_other_position,
    wp::float32 var_other_radius,
    GenericShapeData_ba895271 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    GenericShapeData_ba895271 & adj_shape_data,
    wp::quat_t<wp::float32> & adj_plane_rotation,
    wp::vec_t<3, wp::float32> & adj_plane_position,
    wp::vec_t<3, wp::float32> & adj_other_position,
    wp::float32 & adj_other_radius,
    GenericShapeData_ba895271 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:816
static CUDA_CALLABLE void adj_closest_point_on_triangle_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_tri_a,
    wp::vec_t<3, wp::float32> var_tri_b,
    wp::vec_t<3, wp::float32> var_tri_c,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_tri_a,
    wp::vec_t<3, wp::float32> & adj_tri_b,
    wp::vec_t<3, wp::float32> & adj_tri_c,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE void adj__adjust_minkowski_center_0(
    GenericShapeData_ba895271 var_geom_a,
    wp::vec_t<3, wp::float32> var_center_b_world,
    wp::vec_t<3, wp::float32> var_center_b_to_a,
    GenericShapeData_ba895271 & adj_geom_a,
    wp::vec_t<3, wp::float32> & adj_center_b_world,
    wp::vec_t<3, wp::float32> & adj_center_b_to_a,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE void adj_create_shape_center_function__locals__shape_center_16(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    GenericShapeData_ba895271 & adj_geom_a,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::quat_t<wp::float32> & adj_orientation_b,
    wp::vec_t<3, wp::float32> & adj_position_b,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    MinkowskiCenter_f7efefbe & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE void adj__minkowski_center_fallback_0(
    GenericShapeData_ba895271 var_geom_a,
    wp::vec_t<3, wp::float32> var_center_b_world,
    GenericShapeData_ba895271 & adj_geom_a,
    wp::vec_t<3, wp::float32> & adj_center_b_world,
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:46
static CUDA_CALLABLE void adj__support_rsqrt_rn_0(
    wp::float32 value,
    wp::float32 & adj_value,
    wp::float32 & adj_ret)
{
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE void adj_support_map_lean_0(
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE void adj_create_shape_support_function__locals__shape_support_26(
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE void adj_create_shape_support_function__locals__shape_support_27(
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE void adj_create_support_map_function__locals__support_map_b_17(
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_direction,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::vec_t<3, wp::float32> & adj_direction,
    wp::quat_t<wp::float32> & adj_orientation_b,
    wp::vec_t<3, wp::float32> & adj_position_b,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE void adj_create_support_map_function__locals__minkowski_support_17(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_direction,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::float32 var_extend,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    GenericShapeData_ba895271 & adj_geom_a,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::vec_t<3, wp::float32> & adj_direction,
    wp::quat_t<wp::float32> & adj_orientation_b,
    wp::vec_t<3, wp::float32> & adj_position_b,
    wp::float32 & adj_extend,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    Vert_99873389 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:52
static CUDA_CALLABLE void adj_vert_a_0(
    Vert_99873389 var_vert,
    Vert_99873389 & adj_vert,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE void adj_create_solve_mpr__locals__solve_mpr_core_8(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::float32 var_extend,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    wp::int32 var_MAX_ITER,
    wp::float32 var_COLLIDE_EPSILON,
    bool & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::float32 & ret_4,
    GenericShapeData_ba895271 & adj_geom_a,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::quat_t<wp::float32> & adj_orientation_b,
    wp::vec_t<3, wp::float32> & adj_position_b,
    wp::float32 & adj_extend,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    wp::int32 & adj_MAX_ITER,
    wp::float32 & adj_COLLIDE_EPSILON,
    bool & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2,
    wp::vec_t<3, wp::float32> & adj_ret_3,
    wp::float32 & adj_ret_4)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:77
static CUDA_CALLABLE void adj_create_solve_closest_distance__locals__simplex_get_vertex_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::int32 var_i,
    wp::mat_t<8, 3, wp::float32> & adj_v,
    wp::int32 & adj_i,
    Vert_99873389 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:293
static CUDA_CALLABLE void adj_create_solve_closest_distance__locals__simplex_get_closest_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::vec_t<4, wp::float32> var_barycentric,
    wp::uint32 var_usage_mask,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::mat_t<8, 3, wp::float32> & adj_v,
    wp::vec_t<4, wp::float32> & adj_barycentric,
    wp::uint32 & adj_usage_mask,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/support_function.py:0
static CUDA_CALLABLE void adj_create_shape_support_function__locals__shape_support_25(
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE void adj_create_support_map_function__locals__support_map_b_16(
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_direction,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::vec_t<3, wp::float32> & adj_direction,
    wp::quat_t<wp::float32> & adj_orientation_b,
    wp::vec_t<3, wp::float32> & adj_position_b,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/mpr.py:0
static CUDA_CALLABLE void adj_create_support_map_function__locals__minkowski_support_16(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_direction,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::float32 var_extend,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    GenericShapeData_ba895271 & adj_geom_a,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::vec_t<3, wp::float32> & adj_direction,
    wp::quat_t<wp::float32> & adj_orientation_b,
    wp::vec_t<3, wp::float32> & adj_position_b,
    wp::float32 & adj_extend,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    Vert_99873389 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:91
static CUDA_CALLABLE void adj_create_solve_closest_distance__locals__closest_segment_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::int32 var_i0,
    wp::int32 var_i1,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<4, wp::float32> & ret_1,
    wp::uint32 & ret_2,
    wp::mat_t<8, 3, wp::float32> & adj_v,
    wp::int32 & adj_i0,
    wp::int32 & adj_i1,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<4, wp::float32> & adj_ret_1,
    wp::uint32 & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:134
static CUDA_CALLABLE void adj_create_solve_closest_distance__locals__closest_triangle_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::int32 var_i0,
    wp::int32 var_i1,
    wp::int32 var_i2,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<4, wp::float32> & ret_1,
    wp::uint32 & ret_2,
    wp::mat_t<8, 3, wp::float32> & adj_v,
    wp::int32 & adj_i0,
    wp::int32 & adj_i1,
    wp::int32 & adj_i2,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<4, wp::float32> & adj_ret_1,
    wp::uint32 & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:210
static CUDA_CALLABLE void adj_create_solve_closest_distance__locals__determinant_8(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_c,
    wp::vec_t<3, wp::float32> var_d,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::vec_t<3, wp::float32> & adj_c,
    wp::vec_t<3, wp::float32> & adj_d,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:215
static CUDA_CALLABLE void adj_create_solve_closest_distance__locals__closest_tetrahedron_8(
    wp::mat_t<8, 3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<4, wp::float32> & ret_1,
    wp::uint32 & ret_2,
    wp::mat_t<8, 3, wp::float32> & adj_v,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<4, wp::float32> & adj_ret_1,
    wp::uint32 & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/simplex_solver.py:0
static CUDA_CALLABLE void adj_create_solve_closest_distance__locals__solve_closest_distance_core_8(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::float32 var_extend,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    wp::int32 var_MAX_ITER,
    wp::float32 var_COLLIDE_EPSILON,
    bool & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::float32 & ret_4,
    GenericShapeData_ba895271 & adj_geom_a,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::quat_t<wp::float32> & adj_orientation_b,
    wp::vec_t<3, wp::float32> & adj_position_b,
    wp::float32 & adj_extend,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    wp::int32 & adj_MAX_ITER,
    wp::float32 & adj_COLLIDE_EPSILON,
    bool & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2,
    wp::vec_t<3, wp::float32> & adj_ret_3,
    wp::float32 & adj_ret_4)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:144
static CUDA_CALLABLE void adj_post_process_minkowski_only_0(
    ContactData_40360d7c var_contact_data,
    GenericShapeData_ba895271 var_shape_a,
    wp::vec_t<3, wp::float32> var_pos_a_adjusted,
    wp::quat_t<wp::float32> var_rot_a,
    GenericShapeData_ba895271 var_shape_b,
    wp::vec_t<3, wp::float32> var_pos_b_adjusted,
    wp::quat_t<wp::float32> var_rot_b,
    ContactData_40360d7c & adj_contact_data,
    GenericShapeData_ba895271 & adj_shape_a,
    wp::vec_t<3, wp::float32> & adj_pos_a_adjusted,
    wp::quat_t<wp::float32> & adj_rot_a,
    GenericShapeData_ba895271 & adj_shape_b,
    wp::vec_t<3, wp::float32> & adj_pos_b_adjusted,
    wp::quat_t<wp::float32> & adj_rot_b,
    ContactData_40360d7c & adj_ret)
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/math/__init__.py:235
static CUDA_CALLABLE void adj_orthonormal_basis_0(
    wp::vec_t<3, wp::float32> var_n,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_n,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:623
static CUDA_CALLABLE void adj_add_avoid_duplicates_vec2_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_arr,
    wp::int32 var_arr_count,
    wp::vec_t<2, wp::float32> var_vec,
    wp::float32 var_eps,
    wp::int32 & ret_0,
    bool & ret_1,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_arr,
    wp::int32 & adj_arr_count,
    wp::vec_t<2, wp::float32> & adj_vec,
    wp::float32 & adj_eps,
    wp::int32 & adj_ret_0,
    bool & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:141
static CUDA_CALLABLE void adj_update_incremental_plane_tracker_0(
    IncrementalPlaneTracker_8af6ef33 var_tracker,
    wp::vec_t<3, wp::float32> var_current_point,
    wp::int32 var_current_point_id,
    IncrementalPlaneTracker_8af6ef33 & adj_tracker,
    wp::vec_t<3, wp::float32> & adj_current_point,
    wp::int32 & adj_current_point_id,
    IncrementalPlaneTracker_8af6ef33 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:167
static CUDA_CALLABLE void adj_compute_line_segment_projector_normal_0(
    wp::vec_t<3, wp::float32> var_segment_dir,
    wp::vec_t<3, wp::float32> var_reference_normal,
    wp::vec_t<3, wp::float32> & adj_segment_dir,
    wp::vec_t<3, wp::float32> & adj_reference_normal,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:189
static CUDA_CALLABLE void adj_create_body_projectors_0(
    IncrementalPlaneTracker_8af6ef33 var_plane_tracker_a,
    wp::vec_t<3, wp::float32> var_anchor_point_a,
    IncrementalPlaneTracker_8af6ef33 var_plane_tracker_b,
    wp::vec_t<3, wp::float32> var_anchor_point_b,
    wp::vec_t<3, wp::float32> var_contact_normal,
    BodyProjector_d067bb7a & ret_0,
    BodyProjector_d067bb7a & ret_1,
    IncrementalPlaneTracker_8af6ef33 & adj_plane_tracker_a,
    wp::vec_t<3, wp::float32> & adj_anchor_point_a,
    IncrementalPlaneTracker_8af6ef33 & adj_plane_tracker_b,
    wp::vec_t<3, wp::float32> & adj_anchor_point_b,
    wp::vec_t<3, wp::float32> & adj_contact_normal,
    BodyProjector_d067bb7a & adj_ret_0,
    BodyProjector_d067bb7a & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:43
static CUDA_CALLABLE void adj_excess_normal_deviation_0(
    wp::vec_t<3, wp::float32> var_dir_a,
    wp::vec_t<3, wp::float32> var_dir_b,
    wp::vec_t<3, wp::float32> & adj_dir_a,
    wp::vec_t<3, wp::float32> & adj_dir_b,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:62
static CUDA_CALLABLE void adj_signed_area_0(
    wp::vec_t<2, wp::float32> var_a,
    wp::vec_t<2, wp::float32> var_b,
    wp::vec_t<2, wp::float32> var_query_point,
    wp::vec_t<2, wp::float32> & adj_a,
    wp::vec_t<2, wp::float32> & adj_b,
    wp::vec_t<2, wp::float32> & adj_query_point,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:263
static CUDA_CALLABLE void adj_intersection_point_0(
    wp::vec_t<2, wp::float32> var_trim_seg_start,
    wp::vec_t<2, wp::float32> var_trim_seg_end,
    wp::vec_t<2, wp::float32> var_a,
    wp::vec_t<2, wp::float32> var_b,
    wp::vec_t<2, wp::float32> & adj_trim_seg_start,
    wp::vec_t<2, wp::float32> & adj_trim_seg_end,
    wp::vec_t<2, wp::float32> & adj_a,
    wp::vec_t<2, wp::float32> & adj_b,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:289
static CUDA_CALLABLE void adj_insert_vec2_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_arr,
    wp::int32 var_arr_count,
    wp::int32 var_index,
    wp::vec_t<2, wp::float32> var_element,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_arr,
    wp::int32 & adj_arr_count,
    wp::int32 & adj_index,
    wp::vec_t<2, wp::float32> & adj_element)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:307
static CUDA_CALLABLE void adj_trim_in_place_0(
    wp::vec_t<2, wp::float32> var_trim_seg_start,
    wp::vec_t<2, wp::float32> var_trim_seg_end,
    wp::array_t<wp::vec_t<2, wp::float32>> var_loop,
    wp::int32 var_loop_count,
    wp::vec_t<2, wp::float32> & adj_trim_seg_start,
    wp::vec_t<2, wp::float32> & adj_trim_seg_end,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_loop,
    wp::int32 & adj_loop_count,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:398
static CUDA_CALLABLE void adj_trim_all_in_place_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_trim_poly,
    wp::int32 var_trim_poly_count,
    wp::array_t<wp::vec_t<2, wp::float32>> var_loop,
    wp::int32 var_loop_count,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_trim_poly,
    wp::int32 & adj_trim_poly_count,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_loop,
    wp::int32 & adj_loop_count,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:583
static CUDA_CALLABLE void adj_remove_zero_length_edges_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_loop,
    wp::int32 var_loop_count,
    wp::float32 var_eps,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_loop,
    wp::int32 & adj_loop_count,
    wp::float32 & adj_eps,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:487
static CUDA_CALLABLE void adj_approx_max_quadrilateral_area_with_calipers_0(
    wp::array_t<wp::vec_t<2, wp::float32>> var_hull,
    wp::int32 var_hull_count,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_hull,
    wp::int32 & adj_hull_count,
    wp::vec_t<4, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:87
static CUDA_CALLABLE void adj_ray_plane_intersection_0(
    wp::vec_t<3, wp::float32> var_ray_origin,
    wp::vec_t<3, wp::float32> var_ray_direction,
    wp::float32 var_plane_d,
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> & adj_ray_origin,
    wp::vec_t<3, wp::float32> & adj_ray_direction,
    wp::float32 & adj_plane_d,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:239
static CUDA_CALLABLE void adj_body_projector_project_0(
    BodyProjector_d067bb7a var_proj,
    wp::vec_t<3, wp::float32> var_input,
    wp::vec_t<3, wp::float32> var_contact_normal,
    BodyProjector_d067bb7a & adj_proj,
    wp::vec_t<3, wp::float32> & adj_input,
    wp::vec_t<3, wp::float32> & adj_contact_normal,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:0
static CUDA_CALLABLE void adj_create_build_manifold__locals__extract_4_point_contact_manifolds_5(
    wp::array_t<wp::vec_t<2, wp::float32>> var_m_a,
    wp::int32 var_m_a_count,
    wp::fixedarray_t<10, wp::vec_t<2, wp::float32>> var_m_b,
    wp::int32 var_m_b_count,
    wp::vec_t<3, wp::float32> var_normal_local,
    wp::vec_t<3, wp::float32> var_cross_vector_1,
    wp::vec_t<3, wp::float32> var_cross_vector_2,
    wp::vec_t<3, wp::float32> var_center_local,
    BodyProjector_d067bb7a var_projector_a,
    BodyProjector_d067bb7a var_projector_b,
    wp::quat_t<wp::float32> var_orientation_a,
    wp::vec_t<3, wp::float32> var_position_a_world,
    wp::vec_t<3, wp::float32> var_normal_world,
    ContactWriterData_099b2624 var_writer_data,
    ContactData_40360d7c var_contact_template,
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::vec_t<3, wp::float32> var_position_a,
    wp::vec_t<3, wp::float32> var_position_b,
    wp::quat_t<wp::float32> var_quaternion_a,
    wp::quat_t<wp::float32> var_quaternion_b,
    wp::int32 & ret_0,
    wp::float32 & ret_1,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_m_a,
    wp::int32 & adj_m_a_count,
    wp::fixedarray_t<10, wp::vec_t<2, wp::float32>> & adj_m_b,
    wp::int32 & adj_m_b_count,
    wp::vec_t<3, wp::float32> & adj_normal_local,
    wp::vec_t<3, wp::float32> & adj_cross_vector_1,
    wp::vec_t<3, wp::float32> & adj_cross_vector_2,
    wp::vec_t<3, wp::float32> & adj_center_local,
    BodyProjector_d067bb7a & adj_projector_a,
    BodyProjector_d067bb7a & adj_projector_b,
    wp::quat_t<wp::float32> & adj_orientation_a,
    wp::vec_t<3, wp::float32> & adj_position_a_world,
    wp::vec_t<3, wp::float32> & adj_normal_world,
    ContactWriterData_099b2624 & adj_writer_data,
    ContactData_40360d7c & adj_contact_template,
    GenericShapeData_ba895271 & adj_geom_a,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::vec_t<3, wp::float32> & adj_position_a,
    wp::vec_t<3, wp::float32> & adj_position_b,
    wp::quat_t<wp::float32> & adj_quaternion_a,
    wp::quat_t<wp::float32> & adj_quaternion_b,
    wp::int32 & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:38
static CUDA_CALLABLE void adj_should_include_deepest_contact_0(
    wp::float32 var_normal_dot,
    wp::float32 & adj_normal_dot,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/multicontact.py:0
static CUDA_CALLABLE void adj_create_build_manifold__locals__build_manifold_5(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_a,
    wp::vec_t<3, wp::float32> var_position_a_world,
    wp::quat_t<wp::float32> var_relative_orientation_b,
    wp::vec_t<3, wp::float32> var_relative_position_b,
    wp::vec_t<3, wp::float32> var_p_a,
    wp::vec_t<3, wp::float32> var_p_b,
    wp::vec_t<3, wp::float32> var_normal,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    ContactWriterData_099b2624 var_writer_data,
    ContactData_40360d7c var_contact_template,
    GenericShapeData_ba895271 & adj_geom_a,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::quat_t<wp::float32> & adj_orientation_a,
    wp::vec_t<3, wp::float32> & adj_position_a_world,
    wp::quat_t<wp::float32> & adj_relative_orientation_b,
    wp::vec_t<3, wp::float32> & adj_relative_position_b,
    wp::vec_t<3, wp::float32> & adj_p_a,
    wp::vec_t<3, wp::float32> & adj_p_b,
    wp::vec_t<3, wp::float32> & adj_normal,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    ContactWriterData_099b2624 & adj_writer_data,
    ContactData_40360d7c & adj_contact_template,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_convex.py:0
static CUDA_CALLABLE void adj_create_solve_convex_multi_contact__locals__solve_convex_multi_contact_4(
    GenericShapeData_ba895271 var_geom_a,
    GenericShapeData_ba895271 var_geom_b,
    wp::quat_t<wp::float32> var_orientation_a,
    wp::quat_t<wp::float32> var_orientation_b,
    wp::vec_t<3, wp::float32> var_position_a,
    wp::vec_t<3, wp::float32> var_position_b,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    wp::float32 var_contact_threshold,
    bool var_skip_multi_contact,
    ContactWriterData_099b2624 var_writer_data,
    ContactData_40360d7c var_contact_template,
    GenericShapeData_ba895271 & adj_geom_a,
    GenericShapeData_ba895271 & adj_geom_b,
    wp::quat_t<wp::float32> & adj_orientation_a,
    wp::quat_t<wp::float32> & adj_orientation_b,
    wp::vec_t<3, wp::float32> & adj_position_a,
    wp::vec_t<3, wp::float32> & adj_position_b,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    wp::float32 & adj_contact_threshold,
    bool & adj_skip_multi_contact,
    ContactWriterData_099b2624 & adj_writer_data,
    ContactData_40360d7c & adj_contact_template,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:0
static CUDA_CALLABLE void adj_create_compute_gjk_mpr_contacts__locals__compute_gjk_mpr_contacts_3(
    GenericShapeData_ba895271 var_shape_a_data,
    GenericShapeData_ba895271 var_shape_b_data,
    wp::quat_t<wp::float32> var_rot_a,
    wp::quat_t<wp::float32> var_rot_b,
    wp::vec_t<3, wp::float32> var_pos_a_adjusted,
    wp::vec_t<3, wp::float32> var_pos_b_adjusted,
    wp::float32 var_rigid_gap,
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::float32 var_margin_a,
    wp::float32 var_margin_b,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    ContactWriterData_099b2624 var_writer_data,
    wp::int32 var_sort_sub_key,
    GenericShapeData_ba895271 & adj_shape_a_data,
    GenericShapeData_ba895271 & adj_shape_b_data,
    wp::quat_t<wp::float32> & adj_rot_a,
    wp::quat_t<wp::float32> & adj_rot_b,
    wp::vec_t<3, wp::float32> & adj_pos_a_adjusted,
    wp::vec_t<3, wp::float32> & adj_pos_b_adjusted,
    wp::float32 & adj_rigid_gap,
    wp::int32 & adj_shape_a,
    wp::int32 & adj_shape_b,
    wp::float32 & adj_margin_a,
    wp::float32 & adj_margin_b,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    ContactWriterData_099b2624 & adj_writer_data,
    wp::int32 & adj_sort_sub_key)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/newton/_src/geometry/collision_core.py:0
static CUDA_CALLABLE void adj_create_find_contacts__locals__find_contacts_1(
    wp::vec_t<3, wp::float32> var_pos_a,
    wp::vec_t<3, wp::float32> var_pos_b,
    wp::quat_t<wp::float32> var_quat_a,
    wp::quat_t<wp::float32> var_quat_b,
    GenericShapeData_ba895271 var_shape_data_a,
    GenericShapeData_ba895271 var_shape_data_b,
    bool var_is_infinite_plane_a,
    bool var_is_infinite_plane_b,
    wp::float32 var_bsphere_radius_a,
    wp::float32 var_bsphere_radius_b,
    wp::float32 var_rigid_gap,
    wp::int32 var_shape_a,
    wp::int32 var_shape_b,
    wp::float32 var_margin_a,
    wp::float32 var_margin_b,
    SupportMapDataProvider_e77f8b9f var_data_provider,
    ContactWriterData_099b2624 var_writer_data,
    wp::vec_t<3, wp::float32> & adj_pos_a,
    wp::vec_t<3, wp::float32> & adj_pos_b,
    wp::quat_t<wp::float32> & adj_quat_a,
    wp::quat_t<wp::float32> & adj_quat_b,
    GenericShapeData_ba895271 & adj_shape_data_a,
    GenericShapeData_ba895271 & adj_shape_data_b,
    bool & adj_is_infinite_plane_a,
    bool & adj_is_infinite_plane_b,
    wp::float32 & adj_bsphere_radius_a,
    wp::float32 & adj_bsphere_radius_b,
    wp::float32 & adj_rigid_gap,
    wp::int32 & adj_shape_a,
    wp::int32 & adj_shape_b,
    wp::float32 & adj_margin_a,
    wp::float32 & adj_margin_b,
    SupportMapDataProvider_e77f8b9f & adj_data_provider,
    ContactWriterData_099b2624 & adj_writer_data)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void create_narrow_phase_kernel_gjk_mpr__locals__narrow_phase_kernel_gjk_mpr_6d4b355a_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<2, wp::int32>> var_candidate_pair,
    wp::array_t<wp::int32> var_candidate_pair_count,
    wp::array_t<wp::int32> var_shape_types,
    wp::array_t<wp::vec_t<4, wp::float32>> var_shape_data,
    wp::array_t<wp::transform_t<wp::float32>> var_shape_transform,
    wp::array_t<wp::uint64> var_shape_source,
    wp::array_t<wp::vec_t<4, wp::int32>> var_shape_support_data,
    wp::array_t<wp::int32> var_support_lut,
    wp::array_t<wp::int32> var_support_vertex_offsets,
    wp::array_t<wp::int32> var_support_neighbors,
    wp::array_t<wp::float32> var_shape_gap,
    wp::array_t<wp::float32> var_shape_collision_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_aabb_upper,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_lower,
    wp::array_t<wp::vec_t<3, wp::float32>> var_shape_collision_aabb_upper,
    ContactWriterData_099b2624 var_writer_data,
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
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::range_t var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        bool var_24;
        wp::vec_t<2, wp::int32>* var_25;
        wp::vec_t<2, wp::int32> var_26;
        wp::vec_t<2, wp::int32> var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        bool var_32;
        bool var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        const wp::int32 var_36 = 0;
        bool var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::quat_t<wp::float32> var_45;
        GenericShapeData_ba895271 var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::float32 var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::quat_t<wp::float32> var_50;
        GenericShapeData_ba895271 var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::float32 var_53;
        const wp::int32 var_54 = 10;
        bool var_55;
        const wp::float32 var_56 = 0.5;
        wp::vec_t<3, wp::float32>* var_57;
        wp::vec_t<3, wp::float32>* var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        const wp::int32 var_63 = 10;
        bool var_64;
        const wp::float32 var_65 = 0.5;
        wp::vec_t<3, wp::float32>* var_66;
        wp::vec_t<3, wp::float32>* var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        SupportMapDataProvider_e77f8b9f var_72;
        const bool var_73 = false;
        bool var_74;
        const wp::int32 var_75 = 1;
        bool var_76;
        bool var_77;
        const wp::int32 var_78 = 0;
        wp::float32 var_79;
        const wp::float32 var_80 = 0.0;
        bool var_81;
        const wp::int32 var_82 = 1;
        wp::float32 var_83;
        const wp::float32 var_84 = 0.0;
        bool var_85;
        bool var_86;
        const wp::int32 var_87 = 1;
        bool var_88;
        bool var_89;
        const wp::int32 var_90 = 0;
        wp::float32 var_91;
        const wp::float32 var_92 = 0.0;
        bool var_93;
        const wp::int32 var_94 = 1;
        wp::float32 var_95;
        const wp::float32 var_96 = 0.0;
        bool var_97;
        bool var_98;
        const wp::float32 var_99 = 0.0;
        wp::float32 var_100;
        const wp::float32 var_101 = 0.0;
        wp::float32 var_102;
        bool var_103;
        const bool var_104 = true;
        wp::vec_t<3, wp::float32>* var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<3, wp::float32>* var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32>* var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<3, wp::float32>* var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        const bool var_117 = false;
        wp::vec_t<3, wp::float32> var_118;
        wp::float32 var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        const bool var_124 = false;
        bool var_125;
        bool var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32* var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32* var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        //---------
        // forward
        // def narrow_phase_kernel_gjk_mpr(                                                       <L 1>
        // tid = wp.tid()                                                                         <L 27>
        var_0 = builtin_tid1d();
        // num_work_items = wp.min(candidate_pair.shape[0], candidate_pair_count[0])              <L 29>
        var_1 = &(var_candidate_pair.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_6 = wp::address(var_candidate_pair_count, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::min(var_3, var_8);
        // if num_work_items == 0:                                                                <L 32>
        var_10 = (var_7 == var_9);
        if (var_10) {
            // return                                                                             <L 33>
            continue;
        }
        // block_dim = wp.block_dim()                                                             <L 35>
        var_11 = builtin_block_dim();
        // num_blocks = total_num_threads // block_dim                                            <L 36>
        var_12 = wp::floordiv(var_total_num_threads, var_11);
        // block_index = tid // block_dim                                                         <L 37>
        var_13 = wp::floordiv(var_0, var_11);
        // lane = tid - block_index * block_dim                                                   <L 38>
        var_14 = wp::mul(var_13, var_11);
        var_15 = wp::sub(var_0, var_14);
        // items_per_block = (num_work_items + num_blocks - 1) // num_blocks                      <L 40>
        var_16 = wp::add(var_7, var_12);
        var_18 = wp::sub(var_16, var_17);
        var_19 = wp::floordiv(var_18, var_12);
        // block_start = block_index * items_per_block                                            <L 41>
        var_20 = wp::mul(var_13, var_19);
        // for local_index in range(lane, items_per_block, block_dim):                            <L 43>
        var_21 = wp::range(var_15, var_19, var_11);
        start_for_1:;
            if (iter_cmp(var_21) == 0) goto end_for_1;
            var_22 = wp::iter_next(var_21);
            // t = block_start + local_index                                                      <L 44>
            var_23 = wp::add(var_20, var_22);
            // if t >= num_work_items:                                                            <L 45>
            var_24 = (var_23 >= var_7);
            if (var_24) {
                // continue                                                                       <L 46>
                goto start_for_1;
            }
            // pair = candidate_pair[t]                                                           <L 48>
            var_25 = wp::address(var_candidate_pair, var_23);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            // shape_a = pair[0]                                                                  <L 49>
            var_29 = wp::extract(var_26, var_28);
            // shape_b = pair[1]                                                                  <L 50>
            var_31 = wp::extract(var_26, var_30);
            // if shape_a == shape_b or shape_a < 0 or shape_b < 0:                               <L 53>
            var_33 = (var_29 == var_31);
            var_32 = var_33;
            if (!var_32) {
                var_35 = (var_29 < var_34);
                var_32 = var_32 || var_35;
            }
            if (!var_32) {
                var_37 = (var_31 < var_36);
                var_32 = var_32 || var_37;
            }
            if (var_32) {
                // continue                                                                       <L 54>
                goto start_for_1;
            }
            // type_a = shape_types[shape_a]                                                      <L 56>
            var_38 = wp::address(var_shape_types, var_29);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // type_b = shape_types[shape_b]                                                      <L 57>
            var_41 = wp::address(var_shape_types, var_31);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // pos_a, quat_a, shape_data_a, scale_a, margin_offset_a = extract_shape_data(        <L 60>
            // shape_a, shape_transform, shape_types, shape_data, shape_source                    <L 61>
            extract_shape_data_0(var_29, var_shape_transform, var_shape_types, var_shape_data, var_shape_source, var_44, var_45, var_46, var_47, var_48);
            // pos_b, quat_b, shape_data_b, scale_b, margin_offset_b = extract_shape_data(        <L 63>
            // shape_b, shape_transform, shape_types, shape_data, shape_source                    <L 64>
            extract_shape_data_0(var_31, var_shape_transform, var_shape_types, var_shape_data, var_shape_source, var_49, var_50, var_51, var_52, var_53);
            // if type_a == GeoType.CONVEX_MESH:                                                  <L 66>
            var_55 = (var_39 == var_54);
            if (var_55) {
                // shape_data_a.center = 0.5 * (shape_collision_aabb_lower[shape_a] + shape_collision_aabb_upper[shape_a])       <L 67>
                var_57 = wp::address(var_shape_collision_aabb_lower, var_29);
                var_58 = wp::address(var_shape_collision_aabb_upper, var_29);
                var_60 = wp::load(var_57);
                var_61 = wp::load(var_58);
                var_59 = wp::add(var_60, var_61);
                var_62 = wp::mul(var_56, var_59);
                var_46.center = var_62;
            }
            // if type_b == GeoType.CONVEX_MESH:                                                  <L 68>
            var_64 = (var_42 == var_63);
            if (var_64) {
                // shape_data_b.center = 0.5 * (shape_collision_aabb_lower[shape_b] + shape_collision_aabb_upper[shape_b])       <L 69>
                var_66 = wp::address(var_shape_collision_aabb_lower, var_31);
                var_67 = wp::address(var_shape_collision_aabb_upper, var_31);
                var_69 = wp::load(var_66);
                var_70 = wp::load(var_67);
                var_68 = wp::add(var_69, var_70);
                var_71 = wp::mul(var_65, var_68);
                var_51.center = var_71;
            }
            // data_provider = provider_type()                                                    <L 70>
            var_72 = SupportMapDataProvider_e77f8b9f();
            // if wp.static(accelerated_support):                                                 <L 71>
            // is_infinite_plane_a = (type_a == GeoType.PLANE) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)       <L 78>
            var_76 = (var_39 == var_75);
            var_74 = var_76;
            if (var_74) {
                var_79 = wp::extract(var_47, var_78);
                var_81 = (var_79 == var_80);
                var_77 = var_81;
                if (var_77) {
                    var_83 = wp::extract(var_47, var_82);
                    var_85 = (var_83 == var_84);
                    var_77 = var_77 && var_85;
                }
                var_74 = var_74 && var_77;
            }
            // is_infinite_plane_b = (type_b == GeoType.PLANE) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)       <L 79>
            var_88 = (var_42 == var_87);
            var_86 = var_88;
            if (var_86) {
                var_91 = wp::extract(var_52, var_90);
                var_93 = (var_91 == var_92);
                var_89 = var_93;
                if (var_89) {
                    var_95 = wp::extract(var_52, var_94);
                    var_97 = (var_95 == var_96);
                    var_89 = var_89 && var_97;
                }
                var_86 = var_86 && var_89;
            }
            // if is_infinite_plane_a and is_infinite_plane_b:                                    <L 82>
            var_98 = var_74;
            if (var_98) {
                var_98 = var_98 && var_86;
            }
            if (var_98) {
                // continue                                                                       <L 83>
                goto start_for_1;
            }
            // bsphere_radius_a = float(0.0)                                                      <L 87>
            var_100 = wp::float(var_99);
            // bsphere_radius_b = float(0.0)                                                      <L 88>
            var_102 = wp::float(var_101);
            // has_infinite_plane = is_infinite_plane_a or is_infinite_plane_b                    <L 89>
            var_103 = var_74;
            if (!var_103) {
                var_103 = var_103 || var_86;
            }
            // if has_infinite_plane:                                                             <L 91>
            if (var_103) {
                // if wp.static(external_aabb):                                                   <L 93>
                // aabb_a_lower = shape_aabb_lower[shape_a]                                       <L 94>
                var_105 = wp::address(var_shape_aabb_lower, var_29);
                var_107 = wp::load(var_105);
                var_106 = wp::copy(var_107);
                // aabb_a_upper = shape_aabb_upper[shape_a]                                       <L 95>
                var_108 = wp::address(var_shape_aabb_upper, var_29);
                var_110 = wp::load(var_108);
                var_109 = wp::copy(var_110);
                // aabb_b_lower = shape_aabb_lower[shape_b]                                       <L 96>
                var_111 = wp::address(var_shape_aabb_lower, var_31);
                var_113 = wp::load(var_111);
                var_112 = wp::copy(var_113);
                // aabb_b_upper = shape_aabb_upper[shape_b]                                       <L 97>
                var_114 = wp::address(var_shape_aabb_upper, var_31);
                var_116 = wp::load(var_114);
                var_115 = wp::copy(var_116);
                // if wp.static(not external_aabb):                                               <L 98>
                // bsphere_center_a, bsphere_radius_a = compute_bounding_sphere_from_aabb(aabb_a_lower, aabb_a_upper)       <L 131>
                compute_bounding_sphere_from_aabb_0(var_106, var_109, var_118, var_119);
                // bsphere_center_b, bsphere_radius_b = compute_bounding_sphere_from_aabb(aabb_b_lower, aabb_b_upper)       <L 132>
                compute_bounding_sphere_from_aabb_0(var_112, var_115, var_120, var_121);
                // overlap_radius_a = bsphere_radius_a                                            <L 133>
                var_122 = wp::copy(var_119);
                // overlap_radius_b = bsphere_radius_b                                            <L 134>
                var_123 = wp::copy(var_121);
                // if wp.static(external_aabb and speculative):                                   <L 138>
                // if not check_infinite_plane_bsphere_overlap(                                   <L 145>
                // shape_data_a,                                                                  <L 146>
                // shape_data_b,                                                                  <L 147>
                // pos_a,                                                                         <L 148>
                // pos_b,                                                                         <L 149>
                // quat_a,                                                                        <L 150>
                // quat_b,                                                                        <L 151>
                // bsphere_center_a,                                                              <L 152>
                // bsphere_center_b,                                                              <L 153>
                // overlap_radius_a,                                                              <L 154>
                // overlap_radius_b,                                                              <L 155>
                var_125 = check_infinite_plane_bsphere_overlap_0(var_46, var_51, var_44, var_49, var_45, var_50, var_118, var_120, var_122, var_123);
                var_126 = wp::unot(var_125);
                if (var_126) {
                    // continue                                                                   <L 157>
                    goto start_for_1;
                }
            }
            var_127 = wp::where(var_103, var_119, var_100);
            var_128 = wp::where(var_103, var_121, var_102);
            // gap_a = shape_gap[shape_a]                                                         <L 160>
            var_129 = wp::address(var_shape_gap, var_29);
            var_131 = wp::load(var_129);
            var_130 = wp::copy(var_131);
            // gap_b = shape_gap[shape_b]                                                         <L 161>
            var_132 = wp::address(var_shape_gap, var_31);
            var_134 = wp::load(var_132);
            var_133 = wp::copy(var_134);
            // gap_sum = gap_a + gap_b                                                            <L 162>
            var_135 = wp::add(var_130, var_133);
            // wp.static(                                                                         <L 165>
            // pos_a,                                                                             <L 168>
            // pos_b,                                                                             <L 169>
            // quat_a,                                                                            <L 170>
            // quat_b,                                                                            <L 171>
            // shape_data_a,                                                                      <L 172>
            // shape_data_b,                                                                      <L 173>
            // is_infinite_plane_a,                                                               <L 174>
            // is_infinite_plane_b,                                                               <L 175>
            // bsphere_radius_a,                                                                  <L 176>
            // bsphere_radius_b,                                                                  <L 177>
            // gap_sum,                                                                           <L 178>
            // shape_a,                                                                           <L 179>
            // shape_b,                                                                           <L 180>
            // margin_offset_a,                                                                   <L 181>
            // margin_offset_b,                                                                   <L 182>
            // data_provider,                                                                     <L 183>
            // writer_data,                                                                       <L 184>
            create_find_contacts__locals__find_contacts_1(var_44, var_49, var_45, var_50, var_46, var_51, var_74, var_86, var_127, var_128, var_135, var_29, var_31, var_48, var_53, var_72, var_writer_data);
            goto start_for_1;
        end_for_1:;
    }
}

