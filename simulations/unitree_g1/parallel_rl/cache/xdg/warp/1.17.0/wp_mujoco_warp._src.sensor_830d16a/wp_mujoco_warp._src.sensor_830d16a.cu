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


struct VolumeData_53ac1a2d
{
    wp::vec_t<3, wp::float32> center;
    wp::vec_t<3, wp::float32> half_size;
    wp::array_t<wp::vec_t<3, wp::float32>> oct_aabb;
    wp::array_t<wp::vec_t<8, wp::int32>> oct_child;
    wp::array_t<wp::vec_t<8, wp::float32>> oct_coeff;
    wp::int32 root;
    bool valid;


    VolumeData_53ac1a2d() = default;
    CUDA_CALLABLE VolumeData_53ac1a2d(wp::vec_t<3, wp::float32> const& center,
    wp::vec_t<3, wp::float32> const& half_size = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& oct_aabb = {},
    wp::array_t<wp::vec_t<8, wp::int32>> const& oct_child = {},
    wp::array_t<wp::vec_t<8, wp::float32>> const& oct_coeff = {},
    wp::int32 const& root = {},
    bool const& valid = {})
        : center{center}
        , half_size{half_size}
        , oct_aabb{oct_aabb}
        , oct_child{oct_child}
        , oct_coeff{oct_coeff}
        , root{root}
        , valid{valid}

    {
    }

    CUDA_CALLABLE VolumeData_53ac1a2d& operator += (const VolumeData_53ac1a2d& rhs)
    {    center += rhs.center;
    half_size += rhs.half_size;
    root += rhs.root;

        return *this;}

};

static CUDA_CALLABLE void adj_VolumeData_53ac1a2d(wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::vec_t<8, wp::int32>> const&,
    wp::array_t<wp::vec_t<8, wp::float32>> const&,
    wp::int32 const&,
    bool const&,
    wp::vec_t<3, wp::float32> & adj_center,
    wp::vec_t<3, wp::float32> & adj_half_size,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::int32>> & adj_oct_child,
    wp::array_t<wp::vec_t<8, wp::float32>> & adj_oct_coeff,
    wp::int32 & adj_root,
    bool & adj_valid,
    VolumeData_53ac1a2d & adj_ret)
{
    adj_center += adj_ret.center;
    adj_half_size += adj_ret.half_size;
    adj_oct_aabb = adj_ret.oct_aabb;
    adj_oct_child = adj_ret.oct_child;
    adj_oct_coeff = adj_ret.oct_coeff;
    adj_root += adj_ret.root;
    adj_valid += adj_ret.valid;
}

// Required when compiling adjoints.
CUDA_CALLABLE VolumeData_53ac1a2d add(const VolumeData_53ac1a2d& a, const VolumeData_53ac1a2d& b)
{
    return VolumeData_53ac1a2d();
}

CUDA_CALLABLE void adj_atomic_add(VolumeData_53ac1a2d* p, VolumeData_53ac1a2d t)
{
    wp::adj_atomic_add(&p->center, t.center);
    wp::adj_atomic_add(&p->half_size, t.half_size);
    wp::adj_atomic_add(&p->oct_aabb, t.oct_aabb);
    wp::adj_atomic_add(&p->oct_child, t.oct_child);
    wp::adj_atomic_add(&p->oct_coeff, t.oct_coeff);
    wp::adj_atomic_add(&p->root, t.root);
    wp::adj_atomic_add(&p->valid, t.valid);
}




struct MeshData_52eaa0fa
{
    wp::int32 nmeshface;
    wp::array_t<wp::int32> mesh_vertadr;
    wp::array_t<wp::vec_t<3, wp::float32>> mesh_vert;
    wp::array_t<wp::int32> mesh_faceadr;
    wp::array_t<wp::vec_t<3, wp::int32>> mesh_face;
    wp::int32 data_id;
    wp::vec_t<3, wp::float32> pos;
    wp::mat_t<3, 3, wp::float32> mat;
    wp::vec_t<3, wp::float32> size;
    wp::vec_t<3, wp::float32> pnt;
    wp::vec_t<3, wp::float32> vec;
    bool valid;


    MeshData_52eaa0fa() = default;
    CUDA_CALLABLE MeshData_52eaa0fa(wp::int32 const& nmeshface,
    wp::array_t<wp::int32> const& mesh_vertadr = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& mesh_vert = {},
    wp::array_t<wp::int32> const& mesh_faceadr = {},
    wp::array_t<wp::vec_t<3, wp::int32>> const& mesh_face = {},
    wp::int32 const& data_id = {},
    wp::vec_t<3, wp::float32> const& pos = {},
    wp::mat_t<3, 3, wp::float32> const& mat = {},
    wp::vec_t<3, wp::float32> const& size = {},
    wp::vec_t<3, wp::float32> const& pnt = {},
    wp::vec_t<3, wp::float32> const& vec = {},
    bool const& valid = {})
        : nmeshface{nmeshface}
        , mesh_vertadr{mesh_vertadr}
        , mesh_vert{mesh_vert}
        , mesh_faceadr{mesh_faceadr}
        , mesh_face{mesh_face}
        , data_id{data_id}
        , pos{pos}
        , mat{mat}
        , size{size}
        , pnt{pnt}
        , vec{vec}
        , valid{valid}

    {
    }

    CUDA_CALLABLE MeshData_52eaa0fa& operator += (const MeshData_52eaa0fa& rhs)
    {    nmeshface += rhs.nmeshface;
    data_id += rhs.data_id;
    pos += rhs.pos;
    mat += rhs.mat;
    size += rhs.size;
    pnt += rhs.pnt;
    vec += rhs.vec;

        return *this;}

};

static CUDA_CALLABLE void adj_MeshData_52eaa0fa(wp::int32 const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::vec_t<3, wp::int32>> const&,
    wp::int32 const&,
    wp::vec_t<3, wp::float32> const&,
    wp::mat_t<3, 3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    bool const&,
    wp::int32 & adj_nmeshface,
    wp::array_t<wp::int32> & adj_mesh_vertadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_vert,
    wp::array_t<wp::int32> & adj_mesh_faceadr,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_mesh_face,
    wp::int32 & adj_data_id,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    bool & adj_valid,
    MeshData_52eaa0fa & adj_ret)
{
    adj_nmeshface += adj_ret.nmeshface;
    adj_mesh_vertadr = adj_ret.mesh_vertadr;
    adj_mesh_vert = adj_ret.mesh_vert;
    adj_mesh_faceadr = adj_ret.mesh_faceadr;
    adj_mesh_face = adj_ret.mesh_face;
    adj_data_id += adj_ret.data_id;
    adj_pos += adj_ret.pos;
    adj_mat += adj_ret.mat;
    adj_size += adj_ret.size;
    adj_pnt += adj_ret.pnt;
    adj_vec += adj_ret.vec;
    adj_valid += adj_ret.valid;
}

// Required when compiling adjoints.
CUDA_CALLABLE MeshData_52eaa0fa add(const MeshData_52eaa0fa& a, const MeshData_52eaa0fa& b)
{
    return MeshData_52eaa0fa();
}

CUDA_CALLABLE void adj_atomic_add(MeshData_52eaa0fa* p, MeshData_52eaa0fa t)
{
    wp::adj_atomic_add(&p->nmeshface, t.nmeshface);
    wp::adj_atomic_add(&p->mesh_vertadr, t.mesh_vertadr);
    wp::adj_atomic_add(&p->mesh_vert, t.mesh_vert);
    wp::adj_atomic_add(&p->mesh_faceadr, t.mesh_faceadr);
    wp::adj_atomic_add(&p->mesh_face, t.mesh_face);
    wp::adj_atomic_add(&p->data_id, t.data_id);
    wp::adj_atomic_add(&p->pos, t.pos);
    wp::adj_atomic_add(&p->mat, t.mat);
    wp::adj_atomic_add(&p->size, t.size);
    wp::adj_atomic_add(&p->pnt, t.pnt);
    wp::adj_atomic_add(&p->vec, t.vec);
    wp::adj_atomic_add(&p->valid, t.valid);
}




// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void normalize_with_norm_1(
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
    // norm = wp.length(x)                                                                    <L 2>
    var_0 = wp::length(var_x);
    // if norm == 0.0:                                                                        <L 3>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return x, 0.0                                                                      <L 4>
        ret_0 = var_x;
        ret_1 = var_3;
        return;
    }
    // return x / norm, norm                                                                  <L 5>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:32
static CUDA_CALLABLE void _ray_map_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def _ray_map(pos: wp.vec3, mat: wp.mat33, pnt: wp.vec3, vec: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:       <L 33>
    // matT = wp.transpose(mat)                                                               <L 45>
    var_0 = wp::transpose(var_mat);
    // lpnt = matT @ (pnt - pos)                                                              <L 46>
    var_1 = wp::sub(var_pnt, var_pos);
    var_2 = wp::mul(var_0, var_1);
    // lvec = matT @ vec                                                                      <L 47>
    var_3 = wp::mul(var_0, var_vec);
    // return lpnt, lvec                                                                      <L 49>
    ret_0 = var_2;
    ret_1 = var_3;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:213
static CUDA_CALLABLE void ray_plane_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-15;
    const wp::float32 var_5 = -1e-15;
    bool var_6;
    const wp::float32 var_7 = -1.0;
    wp::vec_t<3, wp::float32> var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.0;
    bool var_16;
    const wp::float32 var_17 = -1.0;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::vec_t<2, wp::float32> var_31;
    bool var_32;
    bool var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    const wp::int32 var_38 = 0;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    bool var_43;
    bool var_44;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    const wp::float32 var_47 = 0.0;
    bool var_48;
    const wp::int32 var_49 = 1;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 1;
    wp::float32 var_53;
    bool var_54;
    const wp::int32 var_55 = 0;
    const wp::int32 var_56 = 2;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    const wp::int32 var_59 = 2;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    const wp::float32 var_65 = -1.0;
    wp::vec_t<3, wp::float32> var_66;
    //---------
    // forward
    // def ray_plane(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 214>
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 217>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_0, var_1);
    // if lvec[2] > -MJ_MINVAL:                                                               <L 220>
    var_3 = wp::extract(var_1, var_2);
    var_6 = (var_3 > var_5);
    if (var_6) {
        // return -1.0, wp.vec3()                                                             <L 221>
        var_8 = wp::vec_t<3, wp::float32>();
        ret_0 = var_7;
        ret_1 = var_8;
        return;
    }
    // x = -lpnt[2] / lvec[2]                                                                 <L 224>
    var_10 = wp::extract(var_0, var_9);
    var_11 = wp::neg(var_10);
    var_13 = wp::extract(var_1, var_12);
    var_14 = wp::div(var_11, var_13);
    // if x < 0.0:                                                                            <L 225>
    var_16 = (var_14 < var_15);
    if (var_16) {
        // return -1.0, wp.vec3()                                                             <L 226>
        var_18 = wp::vec_t<3, wp::float32>();
        ret_0 = var_17;
        ret_1 = var_18;
        return;
    }
    // p = wp.vec2(lpnt[0] + x * lvec[0], lpnt[1] + x * lvec[1])                              <L 228>
    var_20 = wp::extract(var_0, var_19);
    var_22 = wp::extract(var_1, var_21);
    var_23 = wp::mul(var_14, var_22);
    var_24 = wp::add(var_20, var_23);
    var_26 = wp::extract(var_0, var_25);
    var_28 = wp::extract(var_1, var_27);
    var_29 = wp::mul(var_14, var_28);
    var_30 = wp::add(var_26, var_29);
    var_31 = wp::vec_t<2, wp::float32>(var_24, var_30);
    // if (size[0] <= 0.0 or wp.abs(p[0]) <= size[0]) and (size[1] <= 0.0 or wp.abs(p[1]) <= size[1]):       <L 231>
    var_35 = wp::extract(var_size, var_34);
    var_37 = (var_35 <= var_36);
    var_33 = var_37;
    if (!var_33) {
        var_39 = wp::extract(var_31, var_38);
        var_40 = wp::abs(var_39);
        var_42 = wp::extract(var_size, var_41);
        var_43 = (var_40 <= var_42);
        var_33 = var_33 || var_43;
    }
    var_32 = var_33;
    if (var_32) {
        var_46 = wp::extract(var_size, var_45);
        var_48 = (var_46 <= var_47);
        var_44 = var_48;
        if (!var_44) {
            var_50 = wp::extract(var_31, var_49);
            var_51 = wp::abs(var_50);
            var_53 = wp::extract(var_size, var_52);
            var_54 = (var_51 <= var_53);
            var_44 = var_44 || var_54;
        }
        var_32 = var_32 && var_44;
    }
    if (var_32) {
        // return x, wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])                                 <L 232>
        var_57 = wp::extract(var_mat, var_55, var_56);
        var_60 = wp::extract(var_mat, var_58, var_59);
        var_63 = wp::extract(var_mat, var_61, var_62);
        var_64 = wp::vec_t<3, wp::float32>(var_57, var_60, var_63);
        ret_0 = var_14;
        ret_1 = var_64;
        return;
    }
    if (!var_32) {
        // return -1.0, wp.vec3()                                                             <L 234>
        var_66 = wp::vec_t<3, wp::float32>();
        ret_0 = var_65;
        ret_1 = var_66;
        return;
    }
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:105
static CUDA_CALLABLE void _ray_quad_0(
    wp::float32 var_a,
    wp::float32 var_b,
    wp::float32 var_c,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 1e-15;
    bool var_4;
    const wp::float32 var_5 = -1.0;
    const wp::float32 var_6 = -1.0;
    const wp::float32 var_7 = -1.0;
    wp::vec_t<2, wp::float32> var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 1.0;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<2, wp::float32> var_18;
    const wp::float32 var_19 = 0.0;
    bool var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    const wp::float32 var_23 = -1.0;
    //---------
    // forward
    // def _ray_quad(a: float, b: float, c: float) -> Tuple[float, wp.vec2]:                  <L 106>
    // det = b * b - a * c                                                                    <L 108>
    var_0 = wp::mul(var_b, var_b);
    var_1 = wp::mul(var_a, var_c);
    var_2 = wp::sub(var_0, var_1);
    // if det < MJ_MINVAL:                                                                    <L 109>
    var_4 = (var_2 < var_3);
    if (var_4) {
        // return -1.0, wp.vec2(-1.0, -1.0)                                                   <L 110>
        var_8 = wp::vec_t<2, wp::float32>(var_6, var_7);
        ret_0 = var_5;
        ret_1 = var_8;
        return;
    }
    // det = wp.sqrt(det)                                                                     <L 111>
    var_9 = wp::sqrt(var_2);
    // den = safe_div(1.0, a)                                                                 <L 114>
    var_11 = safe_div_2(var_10, var_a);
    // x0 = (-b - det) * den                                                                  <L 115>
    var_12 = wp::neg(var_b);
    var_13 = wp::sub(var_12, var_9);
    var_14 = wp::mul(var_13, var_11);
    // x1 = (-b + det) * den                                                                  <L 116>
    var_15 = wp::neg(var_b);
    var_16 = wp::add(var_15, var_9);
    var_17 = wp::mul(var_16, var_11);
    // x = wp.vec2(x0, x1)                                                                    <L 117>
    var_18 = wp::vec_t<2, wp::float32>(var_14, var_17);
    // if x0 >= 0.0:                                                                          <L 120>
    var_20 = (var_14 >= var_19);
    if (var_20) {
        // return x0, x                                                                       <L 121>
        ret_0 = var_14;
        ret_1 = var_18;
        return;
    }
    if (!var_20) {
        // elif x1 >= 0.0:                                                                    <L 122>
        var_22 = (var_17 >= var_21);
        if (var_22) {
            // return x1, x                                                                   <L 123>
            ret_0 = var_17;
            ret_1 = var_18;
            return;
        }
        if (!var_22) {
            // return -1.0, x                                                                 <L 125>
            ret_0 = var_23;
            ret_1 = var_18;
            return;
        }
    }
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:237
static CUDA_CALLABLE void ray_sphere_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::float32 var_dist_sqr,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<2, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    //---------
    // forward
    // def ray_sphere(pos: wp.vec3, dist_sqr: float, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 238>
    // dif = pnt - pos                                                                        <L 240>
    var_0 = wp::sub(var_pnt, var_pos);
    // a = wp.dot(vec, vec)                                                                   <L 242>
    var_1 = wp::dot(var_vec, var_vec);
    // b = wp.dot(vec, dif)                                                                   <L 243>
    var_2 = wp::dot(var_vec, var_0);
    // c = wp.dot(dif, dif) - dist_sqr                                                        <L 244>
    var_3 = wp::dot(var_0, var_0);
    var_4 = wp::sub(var_3, var_dist_sqr);
    // sol, _ = _ray_quad(a, b, c)                                                            <L 246>
    _ray_quad_0(var_1, var_2, var_4, var_5, var_6);
    // normal = wp.vec3()                                                                     <L 247>
    var_7 = wp::vec_t<3, wp::float32>();
    // if sol >= 0:                                                                           <L 248>
    var_9 = (var_5 >= var_8);
    if (var_9) {
        // s = pnt + vec * sol                                                                <L 249>
        var_10 = wp::mul(var_vec, var_5);
        var_11 = wp::add(var_pnt, var_10);
        // normal = wp.normalize(s - pos)                                                     <L 250>
        var_12 = wp::sub(var_11, var_pos);
        var_13 = wp::normalize(var_12);
    }
    var_14 = wp::where(var_9, var_13, var_7);
    // return sol, normal                                                                     <L 251>
    ret_0 = var_5;
    ret_1 = var_14;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:254
static CUDA_CALLABLE void ray_capsule_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    const wp::float32 var_10 = -1.0;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::float32 var_14 = -1.0;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 0;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    wp::float32 var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    const wp::int32 var_49 = 1;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::vec_t<2, wp::float32> var_55;
    const wp::int32 var_56 = 0;
    bool var_57;
    const wp::float32 var_58 = 0.0;
    bool var_59;
    const wp::int32 var_60 = 2;
    wp::float32 var_61;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::int32 var_67 = 1;
    wp::float32 var_68;
    bool var_69;
    bool var_70;
    const wp::float32 var_71 = 0.0;
    bool var_72;
    bool var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    const wp::int32 var_77 = 0;
    wp::float32 var_78;
    const wp::int32 var_79 = 1;
    wp::float32 var_80;
    const wp::int32 var_81 = 2;
    wp::float32 var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32> var_86;
    const wp::int32 var_87 = 2;
    wp::float32 var_88;
    const wp::int32 var_89 = 2;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::vec_t<2, wp::float32> var_97;
    const wp::int32 var_98 = 0;
    bool var_99;
    wp::float32 var_100;
    const wp::float32 var_101 = 0.0;
    bool var_102;
    const wp::int32 var_103 = 2;
    wp::float32 var_104;
    wp::float32 var_105;
    const wp::int32 var_106 = 2;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    const wp::int32 var_110 = 1;
    wp::float32 var_111;
    bool var_112;
    bool var_113;
    const wp::float32 var_114 = 0.0;
    bool var_115;
    wp::float32 var_116;
    bool var_117;
    wp::float32 var_118;
    const wp::int32 var_119 = 1;
    wp::float32 var_120;
    wp::int32 var_121;
    wp::float32 var_122;
    wp::int32 var_123;
    const wp::int32 var_124 = 1;
    bool var_125;
    wp::float32 var_126;
    const wp::float32 var_127 = 0.0;
    bool var_128;
    const wp::int32 var_129 = 2;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 2;
    wp::float32 var_133;
    wp::float32 var_134;
    wp::float32 var_135;
    const wp::int32 var_136 = 1;
    wp::float32 var_137;
    bool var_138;
    bool var_139;
    const wp::float32 var_140 = 0.0;
    bool var_141;
    wp::float32 var_142;
    bool var_143;
    wp::float32 var_144;
    const wp::int32 var_145 = 1;
    wp::float32 var_146;
    wp::int32 var_147;
    wp::float32 var_148;
    wp::int32 var_149;
    const wp::int32 var_150 = 0;
    wp::float32 var_151;
    const wp::int32 var_152 = 1;
    wp::float32 var_153;
    const wp::int32 var_154 = 2;
    wp::float32 var_155;
    const wp::int32 var_156 = 1;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::vec_t<3, wp::float32> var_159;
    wp::float32 var_160;
    wp::float32 var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::vec_t<2, wp::float32> var_164;
    const wp::int32 var_165 = 0;
    bool var_166;
    wp::float32 var_167;
    const wp::float32 var_168 = 0.0;
    bool var_169;
    const wp::int32 var_170 = 2;
    wp::float32 var_171;
    wp::float32 var_172;
    const wp::int32 var_173 = 2;
    wp::float32 var_174;
    wp::float32 var_175;
    wp::float32 var_176;
    const wp::int32 var_177 = 1;
    wp::float32 var_178;
    wp::float32 var_179;
    bool var_180;
    bool var_181;
    const wp::float32 var_182 = 0.0;
    bool var_183;
    wp::float32 var_184;
    bool var_185;
    wp::float32 var_186;
    const wp::int32 var_187 = -1;
    wp::float32 var_188;
    wp::int32 var_189;
    wp::float32 var_190;
    wp::int32 var_191;
    const wp::int32 var_192 = 1;
    bool var_193;
    wp::float32 var_194;
    const wp::float32 var_195 = 0.0;
    bool var_196;
    const wp::int32 var_197 = 2;
    wp::float32 var_198;
    wp::float32 var_199;
    const wp::int32 var_200 = 2;
    wp::float32 var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    const wp::int32 var_204 = 1;
    wp::float32 var_205;
    wp::float32 var_206;
    bool var_207;
    bool var_208;
    const wp::float32 var_209 = 0.0;
    bool var_210;
    wp::float32 var_211;
    bool var_212;
    wp::float32 var_213;
    const wp::int32 var_214 = -1;
    wp::float32 var_215;
    wp::int32 var_216;
    wp::float32 var_217;
    wp::int32 var_218;
    wp::vec_t<3, wp::float32> var_219;
    const wp::int32 var_220 = 0;
    bool var_221;
    const wp::int32 var_222 = 0;
    wp::float32 var_223;
    const wp::int32 var_224 = 0;
    wp::float32 var_225;
    wp::float32 var_226;
    wp::float32 var_227;
    const wp::int32 var_228 = 0;
    const wp::int32 var_229 = 1;
    wp::float32 var_230;
    const wp::int32 var_231 = 1;
    wp::float32 var_232;
    wp::float32 var_233;
    wp::float32 var_234;
    const wp::int32 var_235 = 1;
    const wp::int32 var_236 = 0;
    bool var_237;
    const wp::float32 var_238 = 0.0;
    const wp::int32 var_239 = 2;
    const wp::int32 var_240 = 2;
    wp::float32 var_241;
    const wp::int32 var_242 = 2;
    wp::float32 var_243;
    wp::float32 var_244;
    wp::float32 var_245;
    const wp::int32 var_246 = 1;
    wp::float32 var_247;
    wp::float32 var_248;
    wp::float32 var_249;
    wp::float32 var_250;
    const wp::int32 var_251 = 2;
    wp::vec_t<3, wp::float32> var_252;
    wp::vec_t<3, wp::float32> var_253;
    wp::vec_t<3, wp::float32> var_254;
    //---------
    // forward
    // def ray_capsule(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 255>
    // ssz = size[0] + size[1]                                                                <L 258>
    var_1 = wp::extract(var_size, var_0);
    var_3 = wp::extract(var_size, var_2);
    var_4 = wp::add(var_1, var_3);
    // dist_sphere, normal_sphere = ray_sphere(pos, ssz * ssz, pnt, vec)                      <L 259>
    var_5 = wp::mul(var_4, var_4);
    ray_sphere_0(var_pos, var_5, var_pnt, var_vec, var_6, var_7);
    // if dist_sphere < 0:                                                                    <L 260>
    var_9 = (var_6 < var_8);
    if (var_9) {
        // return -1.0, wp.vec3()                                                             <L 261>
        var_11 = wp::vec_t<3, wp::float32>();
        ret_0 = var_10;
        ret_1 = var_11;
        return;
    }
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 264>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_12, var_13);
    // x = -1.0                                                                               <L 267>
    // sq_size0 = size[0] * size[0]                                                           <L 270>
    var_16 = wp::extract(var_size, var_15);
    var_18 = wp::extract(var_size, var_17);
    var_19 = wp::mul(var_16, var_18);
    // a = lvec[0] * lvec[0] + lvec[1] * lvec[1]                                              <L 271>
    var_21 = wp::extract(var_13, var_20);
    var_23 = wp::extract(var_13, var_22);
    var_24 = wp::mul(var_21, var_23);
    var_26 = wp::extract(var_13, var_25);
    var_28 = wp::extract(var_13, var_27);
    var_29 = wp::mul(var_26, var_28);
    var_30 = wp::add(var_24, var_29);
    // b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]                                              <L 272>
    var_32 = wp::extract(var_13, var_31);
    var_34 = wp::extract(var_12, var_33);
    var_35 = wp::mul(var_32, var_34);
    var_37 = wp::extract(var_13, var_36);
    var_39 = wp::extract(var_12, var_38);
    var_40 = wp::mul(var_37, var_39);
    var_41 = wp::add(var_35, var_40);
    // c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - sq_size0                                   <L 273>
    var_43 = wp::extract(var_12, var_42);
    var_45 = wp::extract(var_12, var_44);
    var_46 = wp::mul(var_43, var_45);
    var_48 = wp::extract(var_12, var_47);
    var_50 = wp::extract(var_12, var_49);
    var_51 = wp::mul(var_48, var_50);
    var_52 = wp::add(var_46, var_51);
    var_53 = wp::sub(var_52, var_19);
    // sol, xx = _ray_quad(a, b, c)                                                           <L 276>
    _ray_quad_0(var_30, var_41, var_53, var_54, var_55);
    // part = 0  # -1: bottom, 0: cylinder, 1: top                                            <L 277>
    // if sol >= 0.0 and wp.abs(lpnt[2] + sol * lvec[2]) <= size[1]:                          <L 281>
    var_59 = (var_54 >= var_58);
    var_57 = var_59;
    if (var_57) {
        var_61 = wp::extract(var_12, var_60);
        var_63 = wp::extract(var_13, var_62);
        var_64 = wp::mul(var_54, var_63);
        var_65 = wp::add(var_61, var_64);
        var_66 = wp::abs(var_65);
        var_68 = wp::extract(var_size, var_67);
        var_69 = (var_66 <= var_68);
        var_57 = var_57 && var_69;
    }
    if (var_57) {
        // if x < 0.0 or sol < x:                                                             <L 282>
        var_72 = (var_14 < var_71);
        var_70 = var_72;
        if (!var_70) {
            var_73 = (var_54 < var_14);
            var_70 = var_70 || var_73;
        }
        if (var_70) {
            // x = sol                                                                        <L 283>
            var_74 = wp::copy(var_54);
        }
        var_75 = wp::where(var_70, var_74, var_14);
    }
    var_76 = wp::where(var_57, var_75, var_14);
    // ldif = wp.vec3(lpnt[0], lpnt[1], lpnt[2] - size[1])                                    <L 286>
    var_78 = wp::extract(var_12, var_77);
    var_80 = wp::extract(var_12, var_79);
    var_82 = wp::extract(var_12, var_81);
    var_84 = wp::extract(var_size, var_83);
    var_85 = wp::sub(var_82, var_84);
    var_86 = wp::vec_t<3, wp::float32>(var_78, var_80, var_85);
    // a += lvec[2] * lvec[2]                                                                 <L 287>
    var_88 = wp::extract(var_13, var_87);
    var_90 = wp::extract(var_13, var_89);
    var_91 = wp::mul(var_88, var_90);
    var_92 = wp::add(var_30, var_91);
    // b = wp.dot(lvec, ldif)                                                                 <L 288>
    var_93 = wp::dot(var_13, var_86);
    // c = wp.dot(ldif, ldif) - sq_size0                                                      <L 289>
    var_94 = wp::dot(var_86, var_86);
    var_95 = wp::sub(var_94, var_19);
    // _, xx = _ray_quad(a, b, c)                                                             <L 290>
    _ray_quad_0(var_92, var_93, var_95, var_96, var_97);
    // for i in range(2):                                                                     <L 293>
    // if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] >= size[1]:                              <L 294>
    var_100 = wp::extract(var_97, var_98);
    var_102 = (var_100 >= var_101);
    var_99 = var_102;
    if (var_99) {
        var_104 = wp::extract(var_12, var_103);
        var_105 = wp::extract(var_97, var_98);
        var_107 = wp::extract(var_13, var_106);
        var_108 = wp::mul(var_105, var_107);
        var_109 = wp::add(var_104, var_108);
        var_111 = wp::extract(var_size, var_110);
        var_112 = (var_109 >= var_111);
        var_99 = var_99 && var_112;
    }
    if (var_99) {
        // if x < 0.0 or xx[i] < x:                                                           <L 295>
        var_115 = (var_76 < var_114);
        var_113 = var_115;
        if (!var_113) {
            var_116 = wp::extract(var_97, var_98);
            var_117 = (var_116 < var_76);
            var_113 = var_113 || var_117;
        }
        if (var_113) {
            // x = xx[i]                                                                      <L 296>
            var_118 = wp::extract(var_97, var_98);
            // part = 1                                                                       <L 297>
        }
        var_120 = wp::where(var_113, var_118, var_76);
        var_121 = wp::where(var_113, var_119, var_56);
    }
    var_122 = wp::where(var_99, var_120, var_76);
    var_123 = wp::where(var_99, var_121, var_56);
    // if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] >= size[1]:                              <L 294>
    var_126 = wp::extract(var_97, var_124);
    var_128 = (var_126 >= var_127);
    var_125 = var_128;
    if (var_125) {
        var_130 = wp::extract(var_12, var_129);
        var_131 = wp::extract(var_97, var_124);
        var_133 = wp::extract(var_13, var_132);
        var_134 = wp::mul(var_131, var_133);
        var_135 = wp::add(var_130, var_134);
        var_137 = wp::extract(var_size, var_136);
        var_138 = (var_135 >= var_137);
        var_125 = var_125 && var_138;
    }
    if (var_125) {
        // if x < 0.0 or xx[i] < x:                                                           <L 295>
        var_141 = (var_122 < var_140);
        var_139 = var_141;
        if (!var_139) {
            var_142 = wp::extract(var_97, var_124);
            var_143 = (var_142 < var_122);
            var_139 = var_139 || var_143;
        }
        if (var_139) {
            // x = xx[i]                                                                      <L 296>
            var_144 = wp::extract(var_97, var_124);
            // part = 1                                                                       <L 297>
        }
        var_146 = wp::where(var_139, var_144, var_122);
        var_147 = wp::where(var_139, var_145, var_123);
    }
    var_148 = wp::where(var_125, var_146, var_122);
    var_149 = wp::where(var_125, var_147, var_123);
    // ldif = wp.vec3(ldif[0], ldif[1], lpnt[2] + size[1])                                    <L 300>
    var_151 = wp::extract(var_86, var_150);
    var_153 = wp::extract(var_86, var_152);
    var_155 = wp::extract(var_12, var_154);
    var_157 = wp::extract(var_size, var_156);
    var_158 = wp::add(var_155, var_157);
    var_159 = wp::vec_t<3, wp::float32>(var_151, var_153, var_158);
    // b = wp.dot(lvec, ldif)                                                                 <L 301>
    var_160 = wp::dot(var_13, var_159);
    // c = wp.dot(ldif, ldif) - sq_size0                                                      <L 302>
    var_161 = wp::dot(var_159, var_159);
    var_162 = wp::sub(var_161, var_19);
    // _, xx = _ray_quad(a, b, c)                                                             <L 303>
    _ray_quad_0(var_92, var_160, var_162, var_163, var_164);
    // for i in range(2):                                                                     <L 306>
    // if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] <= -size[1]:                             <L 307>
    var_167 = wp::extract(var_164, var_165);
    var_169 = (var_167 >= var_168);
    var_166 = var_169;
    if (var_166) {
        var_171 = wp::extract(var_12, var_170);
        var_172 = wp::extract(var_164, var_165);
        var_174 = wp::extract(var_13, var_173);
        var_175 = wp::mul(var_172, var_174);
        var_176 = wp::add(var_171, var_175);
        var_178 = wp::extract(var_size, var_177);
        var_179 = wp::neg(var_178);
        var_180 = (var_176 <= var_179);
        var_166 = var_166 && var_180;
    }
    if (var_166) {
        // if x < 0.0 or xx[i] < x:                                                           <L 308>
        var_183 = (var_148 < var_182);
        var_181 = var_183;
        if (!var_181) {
            var_184 = wp::extract(var_164, var_165);
            var_185 = (var_184 < var_148);
            var_181 = var_181 || var_185;
        }
        if (var_181) {
            // x = xx[i]                                                                      <L 309>
            var_186 = wp::extract(var_164, var_165);
            // part = -1                                                                      <L 310>
        }
        var_188 = wp::where(var_181, var_186, var_148);
        var_189 = wp::where(var_181, var_187, var_149);
    }
    var_190 = wp::where(var_166, var_188, var_148);
    var_191 = wp::where(var_166, var_189, var_149);
    // if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] <= -size[1]:                             <L 307>
    var_194 = wp::extract(var_164, var_192);
    var_196 = (var_194 >= var_195);
    var_193 = var_196;
    if (var_193) {
        var_198 = wp::extract(var_12, var_197);
        var_199 = wp::extract(var_164, var_192);
        var_201 = wp::extract(var_13, var_200);
        var_202 = wp::mul(var_199, var_201);
        var_203 = wp::add(var_198, var_202);
        var_205 = wp::extract(var_size, var_204);
        var_206 = wp::neg(var_205);
        var_207 = (var_203 <= var_206);
        var_193 = var_193 && var_207;
    }
    if (var_193) {
        // if x < 0.0 or xx[i] < x:                                                           <L 308>
        var_210 = (var_190 < var_209);
        var_208 = var_210;
        if (!var_208) {
            var_211 = wp::extract(var_164, var_192);
            var_212 = (var_211 < var_190);
            var_208 = var_208 || var_212;
        }
        if (var_208) {
            // x = xx[i]                                                                      <L 309>
            var_213 = wp::extract(var_164, var_192);
            // part = -1                                                                      <L 310>
        }
        var_215 = wp::where(var_208, var_213, var_190);
        var_216 = wp::where(var_208, var_214, var_191);
    }
    var_217 = wp::where(var_193, var_215, var_190);
    var_218 = wp::where(var_193, var_216, var_191);
    // normal = wp.vec3()                                                                     <L 312>
    var_219 = wp::vec_t<3, wp::float32>();
    // if x >= 0:                                                                             <L 313>
    var_221 = (var_217 >= var_220);
    if (var_221) {
        // normal[0] = lpnt[0] + lvec[0] * x                                                  <L 314>
        var_223 = wp::extract(var_12, var_222);
        var_225 = wp::extract(var_13, var_224);
        var_226 = wp::mul(var_225, var_217);
        var_227 = wp::add(var_223, var_226);
        wp::assign_inplace(var_219, var_228, var_227);
        // normal[1] = lpnt[1] + lvec[1] * x                                                  <L 315>
        var_230 = wp::extract(var_12, var_229);
        var_232 = wp::extract(var_13, var_231);
        var_233 = wp::mul(var_232, var_217);
        var_234 = wp::add(var_230, var_233);
        wp::assign_inplace(var_219, var_235, var_234);
        // if part == 0:                                                                      <L 316>
        var_237 = (var_218 == var_236);
        if (var_237) {
            // normal[2] = 0.0                                                                <L 317>
            wp::assign_inplace(var_219, var_239, var_238);
        }
        if (!var_237) {
            // normal[2] = lpnt[2] + lvec[2] * x - size[1] * float(part)                      <L 319>
            var_241 = wp::extract(var_12, var_240);
            var_243 = wp::extract(var_13, var_242);
            var_244 = wp::mul(var_243, var_217);
            var_245 = wp::add(var_241, var_244);
            var_247 = wp::extract(var_size, var_246);
            var_248 = wp::float(var_218);
            var_249 = wp::mul(var_247, var_248);
            var_250 = wp::sub(var_245, var_249);
            wp::assign_inplace(var_219, var_251, var_250);
        }
        // normal = wp.normalize(normal)                                                      <L 322>
        var_252 = wp::normalize(var_219);
        // normal = mat @ normal                                                              <L 323>
        var_253 = wp::mul(var_mat, var_252);
    }
    var_254 = wp::where(var_221, var_253, var_219);
    // return x, normal                                                                       <L 325>
    ret_0 = var_217;
    ret_1 = var_254;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:328
static CUDA_CALLABLE void ray_ellipsoid_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::float32 var_2 = 1.0;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 1.0;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 1.0;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    const wp::float32 var_29 = 1.0;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::vec_t<2, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 0;
    bool var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    //---------
    // forward
    // def ray_ellipsoid(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 329>
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 332>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_0, var_1);
    // s = wp.vec3(safe_div(1.0, size[0] * size[0]), safe_div(1.0, size[1] * size[1]), safe_div(1.0, size[2] * size[2]))       <L 335>
    var_4 = wp::extract(var_size, var_3);
    var_6 = wp::extract(var_size, var_5);
    var_7 = wp::mul(var_4, var_6);
    var_8 = safe_div_2(var_2, var_7);
    var_11 = wp::extract(var_size, var_10);
    var_13 = wp::extract(var_size, var_12);
    var_14 = wp::mul(var_11, var_13);
    var_15 = safe_div_2(var_9, var_14);
    var_18 = wp::extract(var_size, var_17);
    var_20 = wp::extract(var_size, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = safe_div_2(var_16, var_21);
    var_23 = wp::vec_t<3, wp::float32>(var_8, var_15, var_22);
    // slvec = wp.cw_mul(s, lvec)                                                             <L 338>
    var_24 = wp::cw_mul(var_23, var_1);
    // a = wp.dot(slvec, lvec)                                                                <L 339>
    var_25 = wp::dot(var_24, var_1);
    // b = wp.dot(slvec, lpnt)                                                                <L 340>
    var_26 = wp::dot(var_24, var_0);
    // c = wp.dot(wp.cw_mul(s, lpnt), lpnt) - 1.0                                             <L 341>
    var_27 = wp::cw_mul(var_23, var_0);
    var_28 = wp::dot(var_27, var_0);
    var_30 = wp::sub(var_28, var_29);
    // sol, _ = _ray_quad(a, b, c)                                                            <L 344>
    _ray_quad_0(var_25, var_26, var_30, var_31, var_32);
    // normal = wp.vec3()                                                                     <L 346>
    var_33 = wp::vec_t<3, wp::float32>();
    // if sol >= 0:                                                                           <L 347>
    var_35 = (var_31 >= var_34);
    if (var_35) {
        // l = lpnt + lvec * sol                                                              <L 349>
        var_36 = wp::mul(var_1, var_31);
        var_37 = wp::add(var_0, var_36);
        // normal = wp.cw_mul(s, l)                                                           <L 352>
        var_38 = wp::cw_mul(var_23, var_37);
        // normal = wp.normalize(normal)                                                      <L 353>
        var_39 = wp::normalize(var_38);
        // normal = mat @ normal                                                              <L 354>
        var_40 = wp::mul(var_mat, var_39);
    }
    var_41 = wp::where(var_35, var_40, var_33);
    // return sol, normal                                                                     <L 356>
    ret_0 = var_31;
    ret_1 = var_41;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:359
static CUDA_CALLABLE void ray_cylinder_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
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
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 0;
    bool var_14;
    const wp::float32 var_15 = -1.0;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    const wp::float32 var_19 = -1.0;
    const wp::int32 var_20 = 0;
    const wp::int32 var_21 = 2;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 1e-15;
    bool var_25;
    const wp::int32 var_26 = -1;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 2;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::float32 var_37 = 0.0;
    bool var_38;
    const wp::int32 var_39 = 0;
    wp::float32 var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::vec_t<2, wp::float32> var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 0;
    wp::float32 var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    wp::float32 var_57;
    bool var_58;
    bool var_59;
    const wp::float32 var_60 = 0.0;
    bool var_61;
    bool var_62;
    wp::float32 var_63;
    wp::int32 var_64;
    wp::float32 var_65;
    wp::int32 var_66;
    wp::float32 var_67;
    wp::int32 var_68;
    wp::float32 var_69;
    wp::int32 var_70;
    const wp::int32 var_71 = 1;
    wp::float32 var_72;
    const wp::int32 var_73 = 1;
    wp::float32 var_74;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::float32 var_81;
    const wp::float32 var_82 = 0.0;
    bool var_83;
    const wp::int32 var_84 = 0;
    wp::float32 var_85;
    const wp::int32 var_86 = 0;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    const wp::int32 var_90 = 1;
    wp::float32 var_91;
    const wp::int32 var_92 = 1;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::vec_t<2, wp::float32> var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 0;
    wp::float32 var_99;
    const wp::int32 var_100 = 0;
    wp::float32 var_101;
    wp::float32 var_102;
    bool var_103;
    bool var_104;
    const wp::float32 var_105 = 0.0;
    bool var_106;
    bool var_107;
    wp::float32 var_108;
    wp::int32 var_109;
    wp::float32 var_110;
    wp::int32 var_111;
    wp::float32 var_112;
    wp::int32 var_113;
    wp::float32 var_114;
    wp::int32 var_115;
    wp::vec_t<2, wp::float32> var_116;
    wp::float32 var_117;
    wp::int32 var_118;
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
    const wp::int32 var_130 = 0;
    wp::float32 var_131;
    const wp::int32 var_132 = 0;
    wp::float32 var_133;
    wp::float32 var_134;
    const wp::int32 var_135 = 1;
    wp::float32 var_136;
    const wp::int32 var_137 = 1;
    wp::float32 var_138;
    wp::float32 var_139;
    wp::float32 var_140;
    const wp::int32 var_141 = 0;
    wp::float32 var_142;
    const wp::int32 var_143 = 0;
    wp::float32 var_144;
    wp::float32 var_145;
    const wp::int32 var_146 = 1;
    wp::float32 var_147;
    const wp::int32 var_148 = 1;
    wp::float32 var_149;
    wp::float32 var_150;
    wp::float32 var_151;
    const wp::int32 var_152 = 0;
    wp::float32 var_153;
    const wp::int32 var_154 = 0;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::vec_t<2, wp::float32> var_159;
    bool var_160;
    const wp::float32 var_161 = 0.0;
    bool var_162;
    const wp::int32 var_163 = 2;
    wp::float32 var_164;
    const wp::int32 var_165 = 2;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    wp::float32 var_169;
    const wp::int32 var_170 = 1;
    wp::float32 var_171;
    bool var_172;
    bool var_173;
    const wp::float32 var_174 = 0.0;
    bool var_175;
    bool var_176;
    wp::float32 var_177;
    const wp::int32 var_178 = 0;
    wp::float32 var_179;
    wp::int32 var_180;
    wp::float32 var_181;
    wp::int32 var_182;
    wp::vec_t<3, wp::float32> var_183;
    const wp::int32 var_184 = 0;
    bool var_185;
    const wp::int32 var_186 = 0;
    bool var_187;
    wp::vec_t<3, wp::float32> var_188;
    wp::vec_t<3, wp::float32> var_189;
    const wp::float32 var_190 = 0.0;
    const wp::int32 var_191 = 2;
    wp::vec_t<3, wp::float32> var_192;
    const wp::float32 var_193 = 0.0;
    const wp::float32 var_194 = 0.0;
    wp::float32 var_195;
    wp::vec_t<3, wp::float32> var_196;
    wp::vec_t<3, wp::float32> var_197;
    wp::vec_t<3, wp::float32> var_198;
    wp::vec_t<3, wp::float32> var_199;
    //---------
    // forward
    // def ray_cylinder(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 360>
    // ssz = size[0] * size[0] + size[1] * size[1]                                            <L 363>
    var_1 = wp::extract(var_size, var_0);
    var_3 = wp::extract(var_size, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_size, var_5);
    var_8 = wp::extract(var_size, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::add(var_4, var_9);
    // dist_sphere, normal_sphere = ray_sphere(pos, ssz, pnt, vec)                            <L 364>
    ray_sphere_0(var_pos, var_10, var_pnt, var_vec, var_11, var_12);
    // if dist_sphere < 0:                                                                    <L 365>
    var_14 = (var_11 < var_13);
    if (var_14) {
        // return -1.0, wp.vec3()                                                             <L 366>
        var_16 = wp::vec_t<3, wp::float32>();
        ret_0 = var_15;
        ret_1 = var_16;
        return;
    }
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 369>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_17, var_18);
    // x = -1.0                                                                               <L 372>
    // part = 0  # -1: bottom, 0: cylinder, 1: top                                            <L 373>
    // if wp.abs(lvec[2]) > MJ_MINVAL:                                                        <L 376>
    var_22 = wp::extract(var_18, var_21);
    var_23 = wp::abs(var_22);
    var_25 = (var_23 > var_24);
    if (var_25) {
        // for side in range(-1, 2, 2):                                                       <L 377>
        // sol = (float(side) * size[1] - lpnt[2]) / lvec[2]                                  <L 379>
        var_27 = wp::float(var_26);
        var_29 = wp::extract(var_size, var_28);
        var_30 = wp::mul(var_27, var_29);
        var_32 = wp::extract(var_17, var_31);
        var_33 = wp::sub(var_30, var_32);
        var_35 = wp::extract(var_18, var_34);
        var_36 = wp::div(var_33, var_35);
        // if sol >= 0.0:                                                                     <L 382>
        var_38 = (var_36 >= var_37);
        if (var_38) {
            // p = wp.vec2(lpnt[0] + sol * lvec[0], lpnt[1] + sol * lvec[1])                  <L 384>
            var_40 = wp::extract(var_17, var_39);
            var_42 = wp::extract(var_18, var_41);
            var_43 = wp::mul(var_36, var_42);
            var_44 = wp::add(var_40, var_43);
            var_46 = wp::extract(var_17, var_45);
            var_48 = wp::extract(var_18, var_47);
            var_49 = wp::mul(var_36, var_48);
            var_50 = wp::add(var_46, var_49);
            var_51 = wp::vec_t<2, wp::float32>(var_44, var_50);
            // if wp.dot(p, p) <= size[0] * size[0]:                                          <L 387>
            var_52 = wp::dot(var_51, var_51);
            var_54 = wp::extract(var_size, var_53);
            var_56 = wp::extract(var_size, var_55);
            var_57 = wp::mul(var_54, var_56);
            var_58 = (var_52 <= var_57);
            if (var_58) {
                // if x < 0.0 or sol < x:                                                     <L 388>
                var_61 = (var_19 < var_60);
                var_59 = var_61;
                if (!var_59) {
                    var_62 = (var_36 < var_19);
                    var_59 = var_59 || var_62;
                }
                if (var_59) {
                    // x = sol                                                                <L 389>
                    var_63 = wp::copy(var_36);
                    // part = side                                                            <L 390>
                    var_64 = wp::copy(var_26);
                }
                var_65 = wp::where(var_59, var_63, var_19);
                var_66 = wp::where(var_59, var_64, var_20);
            }
            var_67 = wp::where(var_58, var_65, var_19);
            var_68 = wp::where(var_58, var_66, var_20);
        }
        var_69 = wp::where(var_38, var_67, var_19);
        var_70 = wp::where(var_38, var_68, var_20);
        // sol = (float(side) * size[1] - lpnt[2]) / lvec[2]                                  <L 379>
        var_72 = wp::float(var_71);
        var_74 = wp::extract(var_size, var_73);
        var_75 = wp::mul(var_72, var_74);
        var_77 = wp::extract(var_17, var_76);
        var_78 = wp::sub(var_75, var_77);
        var_80 = wp::extract(var_18, var_79);
        var_81 = wp::div(var_78, var_80);
        // if sol >= 0.0:                                                                     <L 382>
        var_83 = (var_81 >= var_82);
        if (var_83) {
            // p = wp.vec2(lpnt[0] + sol * lvec[0], lpnt[1] + sol * lvec[1])                  <L 384>
            var_85 = wp::extract(var_17, var_84);
            var_87 = wp::extract(var_18, var_86);
            var_88 = wp::mul(var_81, var_87);
            var_89 = wp::add(var_85, var_88);
            var_91 = wp::extract(var_17, var_90);
            var_93 = wp::extract(var_18, var_92);
            var_94 = wp::mul(var_81, var_93);
            var_95 = wp::add(var_91, var_94);
            var_96 = wp::vec_t<2, wp::float32>(var_89, var_95);
            // if wp.dot(p, p) <= size[0] * size[0]:                                          <L 387>
            var_97 = wp::dot(var_96, var_96);
            var_99 = wp::extract(var_size, var_98);
            var_101 = wp::extract(var_size, var_100);
            var_102 = wp::mul(var_99, var_101);
            var_103 = (var_97 <= var_102);
            if (var_103) {
                // if x < 0.0 or sol < x:                                                     <L 388>
                var_106 = (var_69 < var_105);
                var_104 = var_106;
                if (!var_104) {
                    var_107 = (var_81 < var_69);
                    var_104 = var_104 || var_107;
                }
                if (var_104) {
                    // x = sol                                                                <L 389>
                    var_108 = wp::copy(var_81);
                    // part = side                                                            <L 390>
                    var_109 = wp::copy(var_71);
                }
                var_110 = wp::where(var_104, var_108, var_69);
                var_111 = wp::where(var_104, var_109, var_70);
            }
            var_112 = wp::where(var_103, var_110, var_69);
            var_113 = wp::where(var_103, var_111, var_70);
        }
        var_114 = wp::where(var_83, var_112, var_69);
        var_115 = wp::where(var_83, var_113, var_70);
        var_116 = wp::where(var_83, var_96, var_51);
    }
    var_117 = wp::where(var_25, var_114, var_19);
    var_118 = wp::where(var_25, var_115, var_20);
    // a = lvec[0] * lvec[0] + lvec[1] * lvec[1]                                              <L 393>
    var_120 = wp::extract(var_18, var_119);
    var_122 = wp::extract(var_18, var_121);
    var_123 = wp::mul(var_120, var_122);
    var_125 = wp::extract(var_18, var_124);
    var_127 = wp::extract(var_18, var_126);
    var_128 = wp::mul(var_125, var_127);
    var_129 = wp::add(var_123, var_128);
    // b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]                                              <L 394>
    var_131 = wp::extract(var_18, var_130);
    var_133 = wp::extract(var_17, var_132);
    var_134 = wp::mul(var_131, var_133);
    var_136 = wp::extract(var_18, var_135);
    var_138 = wp::extract(var_17, var_137);
    var_139 = wp::mul(var_136, var_138);
    var_140 = wp::add(var_134, var_139);
    // c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - size[0] * size[0]                          <L 395>
    var_142 = wp::extract(var_17, var_141);
    var_144 = wp::extract(var_17, var_143);
    var_145 = wp::mul(var_142, var_144);
    var_147 = wp::extract(var_17, var_146);
    var_149 = wp::extract(var_17, var_148);
    var_150 = wp::mul(var_147, var_149);
    var_151 = wp::add(var_145, var_150);
    var_153 = wp::extract(var_size, var_152);
    var_155 = wp::extract(var_size, var_154);
    var_156 = wp::mul(var_153, var_155);
    var_157 = wp::sub(var_151, var_156);
    // sol, _ = _ray_quad(a, b, c)                                                            <L 398>
    _ray_quad_0(var_129, var_140, var_157, var_158, var_159);
    // if sol >= 0.0 and wp.abs(lpnt[2] + sol * lvec[2]) <= size[1]:                          <L 401>
    var_162 = (var_158 >= var_161);
    var_160 = var_162;
    if (var_160) {
        var_164 = wp::extract(var_17, var_163);
        var_166 = wp::extract(var_18, var_165);
        var_167 = wp::mul(var_158, var_166);
        var_168 = wp::add(var_164, var_167);
        var_169 = wp::abs(var_168);
        var_171 = wp::extract(var_size, var_170);
        var_172 = (var_169 <= var_171);
        var_160 = var_160 && var_172;
    }
    if (var_160) {
        // if x < 0.0 or sol < x:                                                             <L 402>
        var_175 = (var_117 < var_174);
        var_173 = var_175;
        if (!var_173) {
            var_176 = (var_158 < var_117);
            var_173 = var_173 || var_176;
        }
        if (var_173) {
            // x = sol                                                                        <L 403>
            var_177 = wp::copy(var_158);
            // part = 0                                                                       <L 404>
        }
        var_179 = wp::where(var_173, var_177, var_117);
        var_180 = wp::where(var_173, var_178, var_118);
    }
    var_181 = wp::where(var_160, var_179, var_117);
    var_182 = wp::where(var_160, var_180, var_118);
    // normal = wp.vec3()                                                                     <L 406>
    var_183 = wp::vec_t<3, wp::float32>();
    // if x >= 0:                                                                             <L 407>
    var_185 = (var_181 >= var_184);
    if (var_185) {
        // if part == 0:                                                                      <L 408>
        var_187 = (var_182 == var_186);
        if (var_187) {
            // normal = lpnt + lvec * x                                                       <L 409>
            var_188 = wp::mul(var_18, var_181);
            var_189 = wp::add(var_17, var_188);
            // normal[2] = 0.0                                                                <L 410>
            wp::assign_inplace(var_189, var_191, var_190);
            // normal = wp.normalize(normal)                                                  <L 411>
            var_192 = wp::normalize(var_189);
        }
        if (!var_187) {
            // normal = wp.vec3(0.0, 0.0, float(part))                                        <L 413>
            var_195 = wp::float(var_182);
            var_196 = wp::vec_t<3, wp::float32>(var_193, var_194, var_195);
        }
        var_197 = wp::where(var_187, var_192, var_196);
        // normal = mat @ normal                                                              <L 415>
        var_198 = wp::mul(var_mat, var_197);
    }
    var_199 = wp::where(var_185, var_198, var_183);
    // return x, normal                                                                       <L 417>
    ret_0 = var_181;
    ret_1 = var_199;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:420
static CUDA_CALLABLE void ray_box_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<6, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = -1.0;
    const wp::float32 var_1 = -1.0;
    const wp::float32 var_2 = -1.0;
    const wp::float32 var_3 = -1.0;
    const wp::float32 var_4 = -1.0;
    const wp::float32 var_5 = -1.0;
    wp::vec_t<6, wp::float32> var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    bool var_11;
    const wp::float32 var_12 = -1.0;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::float32 var_16 = -1.0;
    wp::float32 var_17;
    const wp::int32 var_18 = -1;
    const wp::int32 var_19 = -1;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 1e-15;
    bool var_24;
    const wp::int32 var_25 = -1;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.0;
    bool var_34;
    const wp::int32 var_35 = 0;
    bool var_36;
    const wp::int32 var_37 = 1;
    const wp::int32 var_38 = 0;
    wp::int32 var_39;
    const wp::int32 var_40 = 2;
    bool var_41;
    const wp::int32 var_42 = 1;
    const wp::int32 var_43 = 2;
    wp::int32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    bool var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    bool var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    bool var_59;
    bool var_60;
    const wp::float32 var_61 = 0.0;
    bool var_62;
    bool var_63;
    wp::float32 var_64;
    wp::int32 var_65;
    wp::int32 var_66;
    wp::float32 var_67;
    wp::int32 var_68;
    wp::int32 var_69;
    const wp::int32 var_70 = 2;
    wp::int32 var_71;
    const wp::int32 var_72 = 1;
    wp::int32 var_73;
    const wp::int32 var_74 = 2;
    wp::int32 var_75;
    wp::int32 var_76;
    wp::float32 var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    wp::float32 var_80;
    wp::int32 var_81;
    wp::int32 var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    const wp::float32 var_91 = 0.0;
    bool var_92;
    const wp::int32 var_93 = 0;
    bool var_94;
    const wp::int32 var_95 = 1;
    const wp::int32 var_96 = 0;
    wp::int32 var_97;
    const wp::int32 var_98 = 2;
    bool var_99;
    const wp::int32 var_100 = 1;
    const wp::int32 var_101 = 2;
    wp::int32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    bool var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    bool var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    bool var_117;
    bool var_118;
    const wp::float32 var_119 = 0.0;
    bool var_120;
    bool var_121;
    wp::float32 var_122;
    wp::int32 var_123;
    wp::int32 var_124;
    wp::float32 var_125;
    wp::int32 var_126;
    wp::int32 var_127;
    const wp::int32 var_128 = 2;
    wp::int32 var_129;
    const wp::int32 var_130 = 1;
    wp::int32 var_131;
    const wp::int32 var_132 = 2;
    wp::int32 var_133;
    wp::int32 var_134;
    wp::float32 var_135;
    wp::int32 var_136;
    wp::int32 var_137;
    wp::float32 var_138;
    wp::int32 var_139;
    wp::int32 var_140;
    wp::int32 var_141;
    wp::int32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    wp::float32 var_145;
    wp::int32 var_146;
    wp::int32 var_147;
    const wp::int32 var_148 = 1;
    wp::float32 var_149;
    wp::float32 var_150;
    bool var_151;
    const wp::int32 var_152 = -1;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    const wp::float32 var_160 = 0.0;
    bool var_161;
    const wp::int32 var_162 = 0;
    bool var_163;
    const wp::int32 var_164 = 1;
    const wp::int32 var_165 = 0;
    wp::int32 var_166;
    const wp::int32 var_167 = 2;
    bool var_168;
    const wp::int32 var_169 = 1;
    const wp::int32 var_170 = 2;
    wp::int32 var_171;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::float32 var_174;
    wp::float32 var_175;
    wp::float32 var_176;
    wp::float32 var_177;
    wp::float32 var_178;
    wp::float32 var_179;
    bool var_180;
    wp::float32 var_181;
    wp::float32 var_182;
    bool var_183;
    wp::float32 var_184;
    wp::float32 var_185;
    bool var_186;
    bool var_187;
    const wp::float32 var_188 = 0.0;
    bool var_189;
    bool var_190;
    wp::float32 var_191;
    wp::int32 var_192;
    wp::int32 var_193;
    wp::float32 var_194;
    wp::int32 var_195;
    wp::int32 var_196;
    const wp::int32 var_197 = 2;
    wp::int32 var_198;
    const wp::int32 var_199 = 1;
    wp::int32 var_200;
    const wp::int32 var_201 = 2;
    wp::int32 var_202;
    wp::int32 var_203;
    wp::float32 var_204;
    wp::int32 var_205;
    wp::int32 var_206;
    wp::float32 var_207;
    wp::int32 var_208;
    wp::int32 var_209;
    wp::int32 var_210;
    wp::int32 var_211;
    wp::float32 var_212;
    wp::float32 var_213;
    const wp::int32 var_214 = 1;
    wp::float32 var_215;
    wp::float32 var_216;
    wp::float32 var_217;
    wp::float32 var_218;
    wp::float32 var_219;
    wp::float32 var_220;
    wp::float32 var_221;
    const wp::float32 var_222 = 0.0;
    bool var_223;
    const wp::int32 var_224 = 0;
    bool var_225;
    const wp::int32 var_226 = 1;
    const wp::int32 var_227 = 0;
    wp::int32 var_228;
    const wp::int32 var_229 = 2;
    bool var_230;
    const wp::int32 var_231 = 1;
    const wp::int32 var_232 = 2;
    wp::int32 var_233;
    wp::float32 var_234;
    wp::float32 var_235;
    wp::float32 var_236;
    wp::float32 var_237;
    wp::float32 var_238;
    wp::float32 var_239;
    wp::float32 var_240;
    wp::float32 var_241;
    bool var_242;
    wp::float32 var_243;
    wp::float32 var_244;
    bool var_245;
    wp::float32 var_246;
    wp::float32 var_247;
    bool var_248;
    bool var_249;
    const wp::float32 var_250 = 0.0;
    bool var_251;
    bool var_252;
    wp::float32 var_253;
    wp::int32 var_254;
    wp::int32 var_255;
    wp::float32 var_256;
    wp::int32 var_257;
    wp::int32 var_258;
    const wp::int32 var_259 = 2;
    wp::int32 var_260;
    const wp::int32 var_261 = 1;
    wp::int32 var_262;
    const wp::int32 var_263 = 2;
    wp::int32 var_264;
    wp::int32 var_265;
    wp::float32 var_266;
    wp::int32 var_267;
    wp::int32 var_268;
    wp::float32 var_269;
    wp::int32 var_270;
    wp::int32 var_271;
    wp::int32 var_272;
    wp::int32 var_273;
    wp::float32 var_274;
    wp::float32 var_275;
    wp::float32 var_276;
    wp::int32 var_277;
    wp::int32 var_278;
    wp::int32 var_279;
    wp::float32 var_280;
    wp::int32 var_281;
    wp::int32 var_282;
    wp::float32 var_283;
    wp::float32 var_284;
    const wp::int32 var_285 = 2;
    wp::float32 var_286;
    wp::float32 var_287;
    bool var_288;
    const wp::int32 var_289 = -1;
    wp::float32 var_290;
    wp::float32 var_291;
    wp::float32 var_292;
    wp::float32 var_293;
    wp::float32 var_294;
    wp::float32 var_295;
    wp::float32 var_296;
    const wp::float32 var_297 = 0.0;
    bool var_298;
    const wp::int32 var_299 = 0;
    bool var_300;
    const wp::int32 var_301 = 1;
    const wp::int32 var_302 = 0;
    wp::int32 var_303;
    const wp::int32 var_304 = 2;
    bool var_305;
    const wp::int32 var_306 = 1;
    const wp::int32 var_307 = 2;
    wp::int32 var_308;
    wp::float32 var_309;
    wp::float32 var_310;
    wp::float32 var_311;
    wp::float32 var_312;
    wp::float32 var_313;
    wp::float32 var_314;
    wp::float32 var_315;
    wp::float32 var_316;
    bool var_317;
    wp::float32 var_318;
    wp::float32 var_319;
    bool var_320;
    wp::float32 var_321;
    wp::float32 var_322;
    bool var_323;
    bool var_324;
    const wp::float32 var_325 = 0.0;
    bool var_326;
    bool var_327;
    wp::float32 var_328;
    wp::int32 var_329;
    wp::int32 var_330;
    wp::float32 var_331;
    wp::int32 var_332;
    wp::int32 var_333;
    const wp::int32 var_334 = 2;
    wp::int32 var_335;
    const wp::int32 var_336 = 1;
    wp::int32 var_337;
    const wp::int32 var_338 = 2;
    wp::int32 var_339;
    wp::int32 var_340;
    wp::float32 var_341;
    wp::int32 var_342;
    wp::int32 var_343;
    wp::float32 var_344;
    wp::int32 var_345;
    wp::int32 var_346;
    wp::int32 var_347;
    wp::int32 var_348;
    wp::float32 var_349;
    wp::float32 var_350;
    const wp::int32 var_351 = 1;
    wp::float32 var_352;
    wp::float32 var_353;
    wp::float32 var_354;
    wp::float32 var_355;
    wp::float32 var_356;
    wp::float32 var_357;
    wp::float32 var_358;
    const wp::float32 var_359 = 0.0;
    bool var_360;
    const wp::int32 var_361 = 0;
    bool var_362;
    const wp::int32 var_363 = 1;
    const wp::int32 var_364 = 0;
    wp::int32 var_365;
    const wp::int32 var_366 = 2;
    bool var_367;
    const wp::int32 var_368 = 1;
    const wp::int32 var_369 = 2;
    wp::int32 var_370;
    wp::float32 var_371;
    wp::float32 var_372;
    wp::float32 var_373;
    wp::float32 var_374;
    wp::float32 var_375;
    wp::float32 var_376;
    wp::float32 var_377;
    wp::float32 var_378;
    bool var_379;
    wp::float32 var_380;
    wp::float32 var_381;
    bool var_382;
    wp::float32 var_383;
    wp::float32 var_384;
    bool var_385;
    bool var_386;
    const wp::float32 var_387 = 0.0;
    bool var_388;
    bool var_389;
    wp::float32 var_390;
    wp::int32 var_391;
    wp::int32 var_392;
    wp::float32 var_393;
    wp::int32 var_394;
    wp::int32 var_395;
    const wp::int32 var_396 = 2;
    wp::int32 var_397;
    const wp::int32 var_398 = 1;
    wp::int32 var_399;
    const wp::int32 var_400 = 2;
    wp::int32 var_401;
    wp::int32 var_402;
    wp::float32 var_403;
    wp::int32 var_404;
    wp::int32 var_405;
    wp::float32 var_406;
    wp::int32 var_407;
    wp::int32 var_408;
    wp::int32 var_409;
    wp::int32 var_410;
    wp::float32 var_411;
    wp::float32 var_412;
    wp::float32 var_413;
    wp::int32 var_414;
    wp::int32 var_415;
    wp::int32 var_416;
    wp::float32 var_417;
    wp::int32 var_418;
    wp::int32 var_419;
    wp::float32 var_420;
    wp::float32 var_421;
    wp::vec_t<3, wp::float32> var_422;
    const wp::int32 var_423 = 0;
    bool var_424;
    wp::float32 var_425;
    wp::vec_t<3, wp::float32> var_426;
    wp::vec_t<3, wp::float32> var_427;
    //---------
    // forward
    // def ray_box(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, vec6, wp.vec3]:       <L 421>
    // all = vec6(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)                                         <L 423>
    var_6 = wp::vec_t<6, wp::float32>({var_0, var_1, var_2, var_3, var_4, var_5});
    // ssz = wp.dot(size, size)                                                               <L 426>
    var_7 = wp::dot(var_size, var_size);
    // dist_sphere, _ = ray_sphere(pos, ssz, pnt, vec)                                        <L 427>
    ray_sphere_0(var_pos, var_7, var_pnt, var_vec, var_8, var_9);
    // if dist_sphere < 0:                                                                    <L 428>
    var_11 = (var_8 < var_10);
    if (var_11) {
        // return -1.0, all, wp.vec3()                                                        <L 429>
        var_13 = wp::vec_t<3, wp::float32>();
        ret_0 = var_12;
        ret_1 = var_6;
        ret_2 = var_13;
        return;
    }
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 432>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_14, var_15);
    // x = float(-1.0)                                                                        <L 435>
    var_17 = wp::float(var_16);
    // face_side = -1                                                                         <L 436>
    // face_axis = -1                                                                         <L 437>
    // for i in range(3):                                                                     <L 440>
    // if wp.abs(lvec[i]) > MJ_MINVAL:                                                        <L 441>
    var_21 = wp::extract(var_15, var_20);
    var_22 = wp::abs(var_21);
    var_24 = (var_22 > var_23);
    if (var_24) {
        // for side in range(-1, 2, 2):                                                       <L 442>
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 444>
        var_26 = wp::float(var_25);
        var_27 = wp::extract(var_size, var_20);
        var_28 = wp::mul(var_26, var_27);
        var_29 = wp::extract(var_14, var_20);
        var_30 = wp::sub(var_28, var_29);
        var_31 = wp::extract(var_15, var_20);
        var_32 = wp::div(var_30, var_31);
        // if sol >= 0.0:                                                                     <L 447>
        var_34 = (var_32 >= var_33);
        if (var_34) {
            // id0 = 1 if i == 0 else 0                                                       <L 448>
            var_36 = (var_20 == var_35);
            if (var_36) {
            }
            if (!var_36) {
            }
            var_39 = wp::where(var_36, var_37, var_38);
            // id1 = 1 if i == 2 else 2                                                       <L 449>
            var_41 = (var_20 == var_40);
            if (var_41) {
            }
            if (!var_41) {
            }
            var_44 = wp::where(var_41, var_42, var_43);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 452>
            var_45 = wp::extract(var_14, var_39);
            var_46 = wp::extract(var_15, var_39);
            var_47 = wp::mul(var_32, var_46);
            var_48 = wp::add(var_45, var_47);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 453>
            var_49 = wp::extract(var_14, var_44);
            var_50 = wp::extract(var_15, var_44);
            var_51 = wp::mul(var_32, var_50);
            var_52 = wp::add(var_49, var_51);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 456>
            var_54 = wp::abs(var_48);
            var_55 = wp::extract(var_size, var_39);
            var_56 = (var_54 <= var_55);
            var_53 = var_56;
            if (var_53) {
                var_57 = wp::abs(var_52);
                var_58 = wp::extract(var_size, var_44);
                var_59 = (var_57 <= var_58);
                var_53 = var_53 && var_59;
            }
            if (var_53) {
                // if x < 0.0 or sol < x:                                                     <L 458>
                var_62 = (var_17 < var_61);
                var_60 = var_62;
                if (!var_60) {
                    var_63 = (var_32 < var_17);
                    var_60 = var_60 || var_63;
                }
                if (var_60) {
                    // x = sol                                                                <L 459>
                    var_64 = wp::copy(var_32);
                    // face_axis = i                                                          <L 460>
                    var_65 = wp::copy(var_20);
                    // face_side = side                                                       <L 461>
                    var_66 = wp::copy(var_25);
                }
                var_67 = wp::where(var_60, var_64, var_17);
                var_68 = wp::where(var_60, var_66, var_18);
                var_69 = wp::where(var_60, var_65, var_19);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 464>
                var_71 = wp::mul(var_70, var_20);
                var_73 = wp::add(var_25, var_72);
                var_75 = wp::floordiv(var_73, var_74);
                var_76 = wp::add(var_71, var_75);
                wp::assign_inplace(var_6, var_76, var_32);
            }
            var_77 = wp::where(var_53, var_67, var_17);
            var_78 = wp::where(var_53, var_68, var_18);
            var_79 = wp::where(var_53, var_69, var_19);
        }
        var_80 = wp::where(var_34, var_77, var_17);
        var_81 = wp::where(var_34, var_78, var_18);
        var_82 = wp::where(var_34, var_79, var_19);
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 444>
        var_84 = wp::float(var_83);
        var_85 = wp::extract(var_size, var_20);
        var_86 = wp::mul(var_84, var_85);
        var_87 = wp::extract(var_14, var_20);
        var_88 = wp::sub(var_86, var_87);
        var_89 = wp::extract(var_15, var_20);
        var_90 = wp::div(var_88, var_89);
        // if sol >= 0.0:                                                                     <L 447>
        var_92 = (var_90 >= var_91);
        if (var_92) {
            // id0 = 1 if i == 0 else 0                                                       <L 448>
            var_94 = (var_20 == var_93);
            if (var_94) {
            }
            if (!var_94) {
            }
            var_97 = wp::where(var_94, var_95, var_96);
            // id1 = 1 if i == 2 else 2                                                       <L 449>
            var_99 = (var_20 == var_98);
            if (var_99) {
            }
            if (!var_99) {
            }
            var_102 = wp::where(var_99, var_100, var_101);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 452>
            var_103 = wp::extract(var_14, var_97);
            var_104 = wp::extract(var_15, var_97);
            var_105 = wp::mul(var_90, var_104);
            var_106 = wp::add(var_103, var_105);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 453>
            var_107 = wp::extract(var_14, var_102);
            var_108 = wp::extract(var_15, var_102);
            var_109 = wp::mul(var_90, var_108);
            var_110 = wp::add(var_107, var_109);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 456>
            var_112 = wp::abs(var_106);
            var_113 = wp::extract(var_size, var_97);
            var_114 = (var_112 <= var_113);
            var_111 = var_114;
            if (var_111) {
                var_115 = wp::abs(var_110);
                var_116 = wp::extract(var_size, var_102);
                var_117 = (var_115 <= var_116);
                var_111 = var_111 && var_117;
            }
            if (var_111) {
                // if x < 0.0 or sol < x:                                                     <L 458>
                var_120 = (var_80 < var_119);
                var_118 = var_120;
                if (!var_118) {
                    var_121 = (var_90 < var_80);
                    var_118 = var_118 || var_121;
                }
                if (var_118) {
                    // x = sol                                                                <L 459>
                    var_122 = wp::copy(var_90);
                    // face_axis = i                                                          <L 460>
                    var_123 = wp::copy(var_20);
                    // face_side = side                                                       <L 461>
                    var_124 = wp::copy(var_83);
                }
                var_125 = wp::where(var_118, var_122, var_80);
                var_126 = wp::where(var_118, var_124, var_81);
                var_127 = wp::where(var_118, var_123, var_82);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 464>
                var_129 = wp::mul(var_128, var_20);
                var_131 = wp::add(var_83, var_130);
                var_133 = wp::floordiv(var_131, var_132);
                var_134 = wp::add(var_129, var_133);
                wp::assign_inplace(var_6, var_134, var_90);
            }
            var_135 = wp::where(var_111, var_125, var_80);
            var_136 = wp::where(var_111, var_126, var_81);
            var_137 = wp::where(var_111, var_127, var_82);
        }
        var_138 = wp::where(var_92, var_135, var_80);
        var_139 = wp::where(var_92, var_136, var_81);
        var_140 = wp::where(var_92, var_137, var_82);
        var_141 = wp::where(var_92, var_97, var_39);
        var_142 = wp::where(var_92, var_102, var_44);
        var_143 = wp::where(var_92, var_106, var_48);
        var_144 = wp::where(var_92, var_110, var_52);
    }
    var_145 = wp::where(var_24, var_138, var_17);
    var_146 = wp::where(var_24, var_139, var_18);
    var_147 = wp::where(var_24, var_140, var_19);
    // if wp.abs(lvec[i]) > MJ_MINVAL:                                                        <L 441>
    var_149 = wp::extract(var_15, var_148);
    var_150 = wp::abs(var_149);
    var_151 = (var_150 > var_23);
    if (var_151) {
        // for side in range(-1, 2, 2):                                                       <L 442>
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 444>
        var_153 = wp::float(var_152);
        var_154 = wp::extract(var_size, var_148);
        var_155 = wp::mul(var_153, var_154);
        var_156 = wp::extract(var_14, var_148);
        var_157 = wp::sub(var_155, var_156);
        var_158 = wp::extract(var_15, var_148);
        var_159 = wp::div(var_157, var_158);
        // if sol >= 0.0:                                                                     <L 447>
        var_161 = (var_159 >= var_160);
        if (var_161) {
            // id0 = 1 if i == 0 else 0                                                       <L 448>
            var_163 = (var_148 == var_162);
            if (var_163) {
            }
            if (!var_163) {
            }
            var_166 = wp::where(var_163, var_164, var_165);
            // id1 = 1 if i == 2 else 2                                                       <L 449>
            var_168 = (var_148 == var_167);
            if (var_168) {
            }
            if (!var_168) {
            }
            var_171 = wp::where(var_168, var_169, var_170);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 452>
            var_172 = wp::extract(var_14, var_166);
            var_173 = wp::extract(var_15, var_166);
            var_174 = wp::mul(var_159, var_173);
            var_175 = wp::add(var_172, var_174);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 453>
            var_176 = wp::extract(var_14, var_171);
            var_177 = wp::extract(var_15, var_171);
            var_178 = wp::mul(var_159, var_177);
            var_179 = wp::add(var_176, var_178);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 456>
            var_181 = wp::abs(var_175);
            var_182 = wp::extract(var_size, var_166);
            var_183 = (var_181 <= var_182);
            var_180 = var_183;
            if (var_180) {
                var_184 = wp::abs(var_179);
                var_185 = wp::extract(var_size, var_171);
                var_186 = (var_184 <= var_185);
                var_180 = var_180 && var_186;
            }
            if (var_180) {
                // if x < 0.0 or sol < x:                                                     <L 458>
                var_189 = (var_145 < var_188);
                var_187 = var_189;
                if (!var_187) {
                    var_190 = (var_159 < var_145);
                    var_187 = var_187 || var_190;
                }
                if (var_187) {
                    // x = sol                                                                <L 459>
                    var_191 = wp::copy(var_159);
                    // face_axis = i                                                          <L 460>
                    var_192 = wp::copy(var_148);
                    // face_side = side                                                       <L 461>
                    var_193 = wp::copy(var_152);
                }
                var_194 = wp::where(var_187, var_191, var_145);
                var_195 = wp::where(var_187, var_193, var_146);
                var_196 = wp::where(var_187, var_192, var_147);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 464>
                var_198 = wp::mul(var_197, var_148);
                var_200 = wp::add(var_152, var_199);
                var_202 = wp::floordiv(var_200, var_201);
                var_203 = wp::add(var_198, var_202);
                wp::assign_inplace(var_6, var_203, var_159);
            }
            var_204 = wp::where(var_180, var_194, var_145);
            var_205 = wp::where(var_180, var_195, var_146);
            var_206 = wp::where(var_180, var_196, var_147);
        }
        var_207 = wp::where(var_161, var_204, var_145);
        var_208 = wp::where(var_161, var_205, var_146);
        var_209 = wp::where(var_161, var_206, var_147);
        var_210 = wp::where(var_161, var_166, var_141);
        var_211 = wp::where(var_161, var_171, var_142);
        var_212 = wp::where(var_161, var_175, var_143);
        var_213 = wp::where(var_161, var_179, var_144);
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 444>
        var_215 = wp::float(var_214);
        var_216 = wp::extract(var_size, var_148);
        var_217 = wp::mul(var_215, var_216);
        var_218 = wp::extract(var_14, var_148);
        var_219 = wp::sub(var_217, var_218);
        var_220 = wp::extract(var_15, var_148);
        var_221 = wp::div(var_219, var_220);
        // if sol >= 0.0:                                                                     <L 447>
        var_223 = (var_221 >= var_222);
        if (var_223) {
            // id0 = 1 if i == 0 else 0                                                       <L 448>
            var_225 = (var_148 == var_224);
            if (var_225) {
            }
            if (!var_225) {
            }
            var_228 = wp::where(var_225, var_226, var_227);
            // id1 = 1 if i == 2 else 2                                                       <L 449>
            var_230 = (var_148 == var_229);
            if (var_230) {
            }
            if (!var_230) {
            }
            var_233 = wp::where(var_230, var_231, var_232);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 452>
            var_234 = wp::extract(var_14, var_228);
            var_235 = wp::extract(var_15, var_228);
            var_236 = wp::mul(var_221, var_235);
            var_237 = wp::add(var_234, var_236);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 453>
            var_238 = wp::extract(var_14, var_233);
            var_239 = wp::extract(var_15, var_233);
            var_240 = wp::mul(var_221, var_239);
            var_241 = wp::add(var_238, var_240);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 456>
            var_243 = wp::abs(var_237);
            var_244 = wp::extract(var_size, var_228);
            var_245 = (var_243 <= var_244);
            var_242 = var_245;
            if (var_242) {
                var_246 = wp::abs(var_241);
                var_247 = wp::extract(var_size, var_233);
                var_248 = (var_246 <= var_247);
                var_242 = var_242 && var_248;
            }
            if (var_242) {
                // if x < 0.0 or sol < x:                                                     <L 458>
                var_251 = (var_207 < var_250);
                var_249 = var_251;
                if (!var_249) {
                    var_252 = (var_221 < var_207);
                    var_249 = var_249 || var_252;
                }
                if (var_249) {
                    // x = sol                                                                <L 459>
                    var_253 = wp::copy(var_221);
                    // face_axis = i                                                          <L 460>
                    var_254 = wp::copy(var_148);
                    // face_side = side                                                       <L 461>
                    var_255 = wp::copy(var_214);
                }
                var_256 = wp::where(var_249, var_253, var_207);
                var_257 = wp::where(var_249, var_255, var_208);
                var_258 = wp::where(var_249, var_254, var_209);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 464>
                var_260 = wp::mul(var_259, var_148);
                var_262 = wp::add(var_214, var_261);
                var_264 = wp::floordiv(var_262, var_263);
                var_265 = wp::add(var_260, var_264);
                wp::assign_inplace(var_6, var_265, var_221);
            }
            var_266 = wp::where(var_242, var_256, var_207);
            var_267 = wp::where(var_242, var_257, var_208);
            var_268 = wp::where(var_242, var_258, var_209);
        }
        var_269 = wp::where(var_223, var_266, var_207);
        var_270 = wp::where(var_223, var_267, var_208);
        var_271 = wp::where(var_223, var_268, var_209);
        var_272 = wp::where(var_223, var_228, var_210);
        var_273 = wp::where(var_223, var_233, var_211);
        var_274 = wp::where(var_223, var_237, var_212);
        var_275 = wp::where(var_223, var_241, var_213);
    }
    var_276 = wp::where(var_151, var_269, var_145);
    var_277 = wp::where(var_151, var_270, var_146);
    var_278 = wp::where(var_151, var_271, var_147);
    var_279 = wp::where(var_151, var_214, var_83);
    var_280 = wp::where(var_151, var_221, var_90);
    var_281 = wp::where(var_151, var_272, var_141);
    var_282 = wp::where(var_151, var_273, var_142);
    var_283 = wp::where(var_151, var_274, var_143);
    var_284 = wp::where(var_151, var_275, var_144);
    // if wp.abs(lvec[i]) > MJ_MINVAL:                                                        <L 441>
    var_286 = wp::extract(var_15, var_285);
    var_287 = wp::abs(var_286);
    var_288 = (var_287 > var_23);
    if (var_288) {
        // for side in range(-1, 2, 2):                                                       <L 442>
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 444>
        var_290 = wp::float(var_289);
        var_291 = wp::extract(var_size, var_285);
        var_292 = wp::mul(var_290, var_291);
        var_293 = wp::extract(var_14, var_285);
        var_294 = wp::sub(var_292, var_293);
        var_295 = wp::extract(var_15, var_285);
        var_296 = wp::div(var_294, var_295);
        // if sol >= 0.0:                                                                     <L 447>
        var_298 = (var_296 >= var_297);
        if (var_298) {
            // id0 = 1 if i == 0 else 0                                                       <L 448>
            var_300 = (var_285 == var_299);
            if (var_300) {
            }
            if (!var_300) {
            }
            var_303 = wp::where(var_300, var_301, var_302);
            // id1 = 1 if i == 2 else 2                                                       <L 449>
            var_305 = (var_285 == var_304);
            if (var_305) {
            }
            if (!var_305) {
            }
            var_308 = wp::where(var_305, var_306, var_307);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 452>
            var_309 = wp::extract(var_14, var_303);
            var_310 = wp::extract(var_15, var_303);
            var_311 = wp::mul(var_296, var_310);
            var_312 = wp::add(var_309, var_311);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 453>
            var_313 = wp::extract(var_14, var_308);
            var_314 = wp::extract(var_15, var_308);
            var_315 = wp::mul(var_296, var_314);
            var_316 = wp::add(var_313, var_315);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 456>
            var_318 = wp::abs(var_312);
            var_319 = wp::extract(var_size, var_303);
            var_320 = (var_318 <= var_319);
            var_317 = var_320;
            if (var_317) {
                var_321 = wp::abs(var_316);
                var_322 = wp::extract(var_size, var_308);
                var_323 = (var_321 <= var_322);
                var_317 = var_317 && var_323;
            }
            if (var_317) {
                // if x < 0.0 or sol < x:                                                     <L 458>
                var_326 = (var_276 < var_325);
                var_324 = var_326;
                if (!var_324) {
                    var_327 = (var_296 < var_276);
                    var_324 = var_324 || var_327;
                }
                if (var_324) {
                    // x = sol                                                                <L 459>
                    var_328 = wp::copy(var_296);
                    // face_axis = i                                                          <L 460>
                    var_329 = wp::copy(var_285);
                    // face_side = side                                                       <L 461>
                    var_330 = wp::copy(var_289);
                }
                var_331 = wp::where(var_324, var_328, var_276);
                var_332 = wp::where(var_324, var_330, var_277);
                var_333 = wp::where(var_324, var_329, var_278);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 464>
                var_335 = wp::mul(var_334, var_285);
                var_337 = wp::add(var_289, var_336);
                var_339 = wp::floordiv(var_337, var_338);
                var_340 = wp::add(var_335, var_339);
                wp::assign_inplace(var_6, var_340, var_296);
            }
            var_341 = wp::where(var_317, var_331, var_276);
            var_342 = wp::where(var_317, var_332, var_277);
            var_343 = wp::where(var_317, var_333, var_278);
        }
        var_344 = wp::where(var_298, var_341, var_276);
        var_345 = wp::where(var_298, var_342, var_277);
        var_346 = wp::where(var_298, var_343, var_278);
        var_347 = wp::where(var_298, var_303, var_281);
        var_348 = wp::where(var_298, var_308, var_282);
        var_349 = wp::where(var_298, var_312, var_283);
        var_350 = wp::where(var_298, var_316, var_284);
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 444>
        var_352 = wp::float(var_351);
        var_353 = wp::extract(var_size, var_285);
        var_354 = wp::mul(var_352, var_353);
        var_355 = wp::extract(var_14, var_285);
        var_356 = wp::sub(var_354, var_355);
        var_357 = wp::extract(var_15, var_285);
        var_358 = wp::div(var_356, var_357);
        // if sol >= 0.0:                                                                     <L 447>
        var_360 = (var_358 >= var_359);
        if (var_360) {
            // id0 = 1 if i == 0 else 0                                                       <L 448>
            var_362 = (var_285 == var_361);
            if (var_362) {
            }
            if (!var_362) {
            }
            var_365 = wp::where(var_362, var_363, var_364);
            // id1 = 1 if i == 2 else 2                                                       <L 449>
            var_367 = (var_285 == var_366);
            if (var_367) {
            }
            if (!var_367) {
            }
            var_370 = wp::where(var_367, var_368, var_369);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 452>
            var_371 = wp::extract(var_14, var_365);
            var_372 = wp::extract(var_15, var_365);
            var_373 = wp::mul(var_358, var_372);
            var_374 = wp::add(var_371, var_373);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 453>
            var_375 = wp::extract(var_14, var_370);
            var_376 = wp::extract(var_15, var_370);
            var_377 = wp::mul(var_358, var_376);
            var_378 = wp::add(var_375, var_377);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 456>
            var_380 = wp::abs(var_374);
            var_381 = wp::extract(var_size, var_365);
            var_382 = (var_380 <= var_381);
            var_379 = var_382;
            if (var_379) {
                var_383 = wp::abs(var_378);
                var_384 = wp::extract(var_size, var_370);
                var_385 = (var_383 <= var_384);
                var_379 = var_379 && var_385;
            }
            if (var_379) {
                // if x < 0.0 or sol < x:                                                     <L 458>
                var_388 = (var_344 < var_387);
                var_386 = var_388;
                if (!var_386) {
                    var_389 = (var_358 < var_344);
                    var_386 = var_386 || var_389;
                }
                if (var_386) {
                    // x = sol                                                                <L 459>
                    var_390 = wp::copy(var_358);
                    // face_axis = i                                                          <L 460>
                    var_391 = wp::copy(var_285);
                    // face_side = side                                                       <L 461>
                    var_392 = wp::copy(var_351);
                }
                var_393 = wp::where(var_386, var_390, var_344);
                var_394 = wp::where(var_386, var_392, var_345);
                var_395 = wp::where(var_386, var_391, var_346);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 464>
                var_397 = wp::mul(var_396, var_285);
                var_399 = wp::add(var_351, var_398);
                var_401 = wp::floordiv(var_399, var_400);
                var_402 = wp::add(var_397, var_401);
                wp::assign_inplace(var_6, var_402, var_358);
            }
            var_403 = wp::where(var_379, var_393, var_344);
            var_404 = wp::where(var_379, var_394, var_345);
            var_405 = wp::where(var_379, var_395, var_346);
        }
        var_406 = wp::where(var_360, var_403, var_344);
        var_407 = wp::where(var_360, var_404, var_345);
        var_408 = wp::where(var_360, var_405, var_346);
        var_409 = wp::where(var_360, var_365, var_347);
        var_410 = wp::where(var_360, var_370, var_348);
        var_411 = wp::where(var_360, var_374, var_349);
        var_412 = wp::where(var_360, var_378, var_350);
    }
    var_413 = wp::where(var_288, var_406, var_276);
    var_414 = wp::where(var_288, var_407, var_277);
    var_415 = wp::where(var_288, var_408, var_278);
    var_416 = wp::where(var_288, var_351, var_279);
    var_417 = wp::where(var_288, var_358, var_280);
    var_418 = wp::where(var_288, var_409, var_281);
    var_419 = wp::where(var_288, var_410, var_282);
    var_420 = wp::where(var_288, var_411, var_283);
    var_421 = wp::where(var_288, var_412, var_284);
    // normal = wp.vec3()                                                                     <L 466>
    var_422 = wp::vec_t<3, wp::float32>();
    // if x >= 0:                                                                             <L 467>
    var_424 = (var_413 >= var_423);
    if (var_424) {
        // normal[face_axis] = float(face_side)                                               <L 468>
        var_425 = wp::float(var_414);
        wp::assign_inplace(var_422, var_415, var_425);
        // normal = mat @ normal                                                              <L 469>
        var_426 = wp::mul(var_mat, var_422);
    }
    var_427 = wp::where(var_424, var_426, var_422);
    // return x, all, normal                                                                  <L 471>
    ret_0 = var_413;
    ret_1 = var_6;
    ret_2 = var_427;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:798
static CUDA_CALLABLE void ray_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_geomtype,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    const wp::int32 var_4 = 2;
    bool var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 3;
    bool var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    const wp::int32 var_17 = 4;
    bool var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 5;
    bool var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::int32 var_25 = 6;
    bool var_26;
    wp::float32 var_27;
    wp::vec_t<6, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    const wp::float32 var_30 = -1.0;
    wp::vec_t<3, wp::float32> var_31;
    //---------
    // forward
    // def ray_geom(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3, geomtype: int) -> Tuple[float, wp.vec3]:       <L 799>
    // if geomtype == GeomType.PLANE:                                                         <L 805>
    var_1 = (var_geomtype == var_0);
    if (var_1) {
        // return ray_plane(pos, mat, size, pnt, vec)                                         <L 806>
        ray_plane_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_2, var_3);
        ret_0 = var_2;
        ret_1 = var_3;
        return;
    }
    if (!var_1) {
        // elif geomtype == GeomType.SPHERE:                                                  <L 807>
        var_5 = (var_geomtype == var_4);
        if (var_5) {
            // return ray_sphere(pos, size[0] * size[0], pnt, vec)                            <L 808>
            var_7 = wp::extract(var_size, var_6);
            var_9 = wp::extract(var_size, var_8);
            var_10 = wp::mul(var_7, var_9);
            ray_sphere_0(var_pos, var_10, var_pnt, var_vec, var_11, var_12);
            ret_0 = var_11;
            ret_1 = var_12;
            return;
        }
        if (!var_5) {
            // elif geomtype == GeomType.CAPSULE:                                             <L 809>
            var_14 = (var_geomtype == var_13);
            if (var_14) {
                // return ray_capsule(pos, mat, size, pnt, vec)                               <L 810>
                ray_capsule_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_15, var_16);
                ret_0 = var_15;
                ret_1 = var_16;
                return;
            }
            if (!var_14) {
                // elif geomtype == GeomType.ELLIPSOID:                                       <L 811>
                var_18 = (var_geomtype == var_17);
                if (var_18) {
                    // return ray_ellipsoid(pos, mat, size, pnt, vec)                         <L 812>
                    ray_ellipsoid_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_19, var_20);
                    ret_0 = var_19;
                    ret_1 = var_20;
                    return;
                }
                if (!var_18) {
                    // elif geomtype == GeomType.CYLINDER:                                    <L 813>
                    var_22 = (var_geomtype == var_21);
                    if (var_22) {
                        // return ray_cylinder(pos, mat, size, pnt, vec)                      <L 814>
                        ray_cylinder_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_23, var_24);
                        ret_0 = var_23;
                        ret_1 = var_24;
                        return;
                    }
                    if (!var_22) {
                        // elif geomtype == GeomType.BOX:                                     <L 815>
                        var_26 = (var_geomtype == var_25);
                        if (var_26) {
                            // dist, _, normal = ray_box(pos, mat, size, pnt, vec)            <L 816>
                            ray_box_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_27, var_28, var_29);
                            // return dist, normal                                            <L 817>
                            ret_0 = var_27;
                            ret_1 = var_29;
                            return;
                        }
                        if (!var_26) {
                            // return -1.0, wp.vec3()                                         <L 819>
                            var_31 = wp::vec_t<3, wp::float32>();
                            ret_0 = var_30;
                            ret_1 = var_31;
                            return;
                        }
                    }
                }
            }
        }
    }
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_scalar_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::float32 var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::int32* var_9;
    const wp::int32 var_10 = 41;
    const wp::int32 var_11 = 41;
    wp::int32 var_12;
    bool var_13;
    wp::int32 var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    bool var_24;
    wp::float32 var_25;
    //---------
    // forward
    // def _write_scalar(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 13>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 14>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 16>
    var_8 = (var_4 > var_7);
    var_6 = var_8;
    if (var_6) {
        var_9 = wp::address(var_sensor_type, var_sensorid);
        var_12 = wp::int(var_11);
        var_14 = wp::load(var_9);
        var_13 = (var_14 == var_12);
        var_15 = wp::unot(var_13);
        var_6 = var_6 && var_15;
    }
    if (var_6) {
        // datatype = sensor_datatype[sensorid]                                               <L 17>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 18>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // out[adr] = wp.clamp(sensor, -cutoff, cutoff)                                   <L 19>
            var_21 = wp::neg(var_4);
            var_22 = wp::clamp(var_sensor, var_21, var_4);
            wp::array_store(var_out, var_1, var_22);
            // return                                                                         <L 20>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 21>
            var_24 = (var_17 == var_23);
            if (var_24) {
                // out[adr] = wp.min(sensor, cutoff)                                          <L 22>
                var_25 = wp::min(var_sensor, var_4);
                wp::array_store(var_out, var_1, var_25);
                // return                                                                     <L 23>
                return;
            }
        }
    }
    // out[adr] = sensor                                                                      <L 25>
    wp::array_store(var_out, var_1, var_sensor);
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:726
static CUDA_CALLABLE wp::float32 poly_potential_0(
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
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.5;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.3333333333333333;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.25;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    //---------
    // forward
    // def poly_potential(linear: float, poly: wp.vec2, x: float, flg_odd: int) -> float:       <L 727>
    // x_val = wp.where(flg_odd == 1, wp.abs(x), x)                                           <L 728>
    var_1 = (var_flg_odd == var_0);
    var_2 = wp::abs(var_x);
    var_3 = wp::where(var_1, var_2, var_x);
    // x_val2 = x_val * x_val                                                                 <L 729>
    var_4 = wp::mul(var_3, var_3);
    // x_val3 = x_val2 * x_val                                                                <L 730>
    var_5 = wp::mul(var_4, var_3);
    // x_val4 = x_val3 * x_val                                                                <L 731>
    var_6 = wp::mul(var_5, var_3);
    // res = 0.5 * linear * x_val2                                                            <L 733>
    var_8 = wp::mul(var_7, var_linear);
    var_9 = wp::mul(var_8, var_4);
    // res += poly[0] * wp.static(1.0 / 3.0) * x_val3                                         <L 734>
    var_11 = wp::extract(var_poly, var_10);
    var_13 = wp::mul(var_11, var_12);
    var_14 = wp::mul(var_13, var_5);
    var_15 = wp::add(var_9, var_14);
    // res += poly[1] * 0.25 * x_val4                                                         <L 735>
    var_17 = wp::extract(var_poly, var_16);
    var_19 = wp::mul(var_17, var_18);
    var_20 = wp::mul(var_19, var_6);
    var_21 = wp::add(var_15, var_20);
    // return res                                                                             <L 736>
    return var_21;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:44
static CUDA_CALLABLE wp::vec_t<3, wp::float32> rot_vec_quat_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::quat_t<wp::float32> var_quat)
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
    wp::vec_t<3, wp::float32> var_8;
    const wp::float32 var_9 = 2.0;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    const wp::float32 var_18 = 2.0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    //---------
    // forward
    // def rot_vec_quat(vec: wp.vec3, quat: wp.quat) -> wp.vec3:                              <L 45>
    // s, u = quat[0], wp.vec3(quat[1], quat[2], quat[3])                                     <L 46>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_7 = wp::extract(var_quat, var_6);
    var_8 = wp::vec_t<3, wp::float32>(var_3, var_5, var_7);
    // r = 2.0 * (wp.dot(u, vec) * u) + (s * s - wp.dot(u, u)) * vec                          <L 47>
    var_10 = wp::dot(var_8, var_vec);
    var_11 = wp::mul(var_10, var_8);
    var_12 = wp::mul(var_9, var_11);
    var_13 = wp::mul(var_1, var_1);
    var_14 = wp::dot(var_8, var_8);
    var_15 = wp::sub(var_13, var_14);
    var_16 = wp::mul(var_15, var_vec);
    var_17 = wp::add(var_12, var_16);
    // r = r + 2.0 * s * wp.cross(u, vec)                                                     <L 48>
    var_19 = wp::mul(var_18, var_1);
    var_20 = wp::cross(var_8, var_vec);
    var_21 = wp::mul(var_19, var_20);
    var_22 = wp::add(var_17, var_21);
    // return r                                                                               <L 49>
    return var_22;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:81
static CUDA_CALLABLE void get_sdf_params_0(
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::float32>> var_oct_coeff,
    wp::array_t<wp::int32> var_mesh_octadr,
    wp::array_t<wp::int32> var_plugin,
    wp::array_t<wp::vec_t<128, wp::float32>> var_plugin_attr,
    wp::int32 var_g_type,
    wp::vec_t<3, wp::float32> var_g_size,
    wp::int32 var_plugin_id,
    wp::int32 var_mesh_id,
    wp::vec_t<128, wp::float32> & ret_0,
    wp::int32 & ret_1,
    VolumeData_53ac1a2d & ret_2,
    MeshData_52eaa0fa & ret_3)
{
    //---------
    // primal vars
    wp::vec_t<128, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    const wp::int32 var_9 = 2;
    const wp::int32 var_10 = -1;
    VolumeData_53ac1a2d var_11;
    bool var_12;
    const wp::int32 var_13 = 8;
    bool var_14;
    const wp::int32 var_15 = -1;
    bool var_16;
    wp::vec_t<128, wp::float32>* var_17;
    wp::vec_t<128, wp::float32> var_18;
    wp::vec_t<128, wp::float32> var_19;
    wp::int32* var_20;
    wp::int32 var_21;
    wp::int32 var_22;
    bool var_23;
    const wp::int32 var_24 = 8;
    bool var_25;
    const wp::int32 var_26 = -1;
    bool var_27;
    wp::int32* var_28;
    wp::int32 var_29;
    wp::int32 var_30;
    const wp::int32 var_31 = 0;
    wp::vec_t<3, wp::float32>* var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 1;
    wp::vec_t<3, wp::float32>* var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    const bool var_39 = true;
    bool var_40;
    const wp::int32 var_41 = 7;
    bool var_42;
    const wp::int32 var_43 = -1;
    bool var_44;
    wp::int32* var_45;
    const wp::int32 var_46 = -1;
    bool var_47;
    wp::int32 var_48;
    wp::int32* var_49;
    wp::int32 var_50;
    wp::int32 var_51;
    const wp::int32 var_52 = 0;
    wp::vec_t<3, wp::float32>* var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    const wp::int32 var_56 = 1;
    wp::vec_t<3, wp::float32>* var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32> var_59;
    const bool var_60 = true;
    wp::int32 var_61;
    wp::vec_t<128, wp::float32> var_62;
    wp::int32 var_63;
    MeshData_52eaa0fa var_64;
    //---------
    // forward
    // def get_sdf_params(                                                                    <L 82>
    // attributes = vec_pluginattr()                                                          <L 97>
    var_0 = wp::vec_t<128, wp::float32>();
    // attributes[0] = g_size[0]                                                              <L 98>
    var_2 = wp::extract(var_g_size, var_1);
    wp::assign_inplace(var_0, var_3, var_2);
    // attributes[1] = g_size[1]                                                              <L 99>
    var_5 = wp::extract(var_g_size, var_4);
    wp::assign_inplace(var_0, var_6, var_5);
    // attributes[2] = g_size[2]                                                              <L 100>
    var_8 = wp::extract(var_g_size, var_7);
    wp::assign_inplace(var_0, var_9, var_8);
    // plugin_index = -1                                                                      <L 101>
    // volume_data = VolumeData()                                                             <L 102>
    var_11 = VolumeData_53ac1a2d();
    // if g_type == GeomType.SDF and plugin_id != -1:                                         <L 104>
    var_14 = (var_g_type == var_13);
    var_12 = var_14;
    if (var_12) {
        var_16 = (var_plugin_id != var_15);
        var_12 = var_12 && var_16;
    }
    if (var_12) {
        // attributes = plugin_attr[plugin_id]                                                <L 105>
        var_17 = wp::address(var_plugin_attr, var_plugin_id);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // plugin_index = plugin[plugin_id]                                                   <L 106>
        var_20 = wp::address(var_plugin, var_plugin_id);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
    }
    if (!var_12) {
        // elif g_type == GeomType.SDF and mesh_id != -1:                                     <L 108>
        var_25 = (var_g_type == var_24);
        var_23 = var_25;
        if (var_23) {
            var_27 = (var_mesh_id != var_26);
            var_23 = var_23 && var_27;
        }
        if (var_23) {
            // octadr = mesh_octadr[mesh_id]                                                  <L 109>
            var_28 = wp::address(var_mesh_octadr, var_mesh_id);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // volume_data.center = oct_aabb[octadr, 0]                                       <L 110>
            var_32 = wp::address(var_oct_aabb, var_29, var_31);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            var_11.center = var_33;
            // volume_data.half_size = oct_aabb[octadr, 1]                                    <L 111>
            var_36 = wp::address(var_oct_aabb, var_29, var_35);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            var_11.half_size = var_37;
            // volume_data.root = octadr                                                      <L 112>
            var_11.root = var_29;
            // volume_data.oct_aabb = oct_aabb                                                <L 113>
            var_11.oct_aabb = var_oct_aabb;
            // volume_data.oct_child = oct_child                                              <L 114>
            var_11.oct_child = var_oct_child;
            // volume_data.oct_coeff = oct_coeff                                              <L 115>
            var_11.oct_coeff = var_oct_coeff;
            // volume_data.valid = True                                                       <L 116>
            var_11.valid = var_39;
        }
        if (!var_23) {
            // elif g_type == GeomType.MESH and mesh_id != -1 and mesh_octadr[mesh_id] != -1:       <L 118>
            var_42 = (var_g_type == var_41);
            var_40 = var_42;
            if (var_40) {
                var_44 = (var_mesh_id != var_43);
                var_40 = var_40 && var_44;
            }
            if (var_40) {
                var_45 = wp::address(var_mesh_octadr, var_mesh_id);
                var_48 = wp::load(var_45);
                var_47 = (var_48 != var_46);
                var_40 = var_40 && var_47;
            }
            if (var_40) {
                // octadr = mesh_octadr[mesh_id]                                              <L 119>
                var_49 = wp::address(var_mesh_octadr, var_mesh_id);
                var_51 = wp::load(var_49);
                var_50 = wp::copy(var_51);
                // volume_data.center = oct_aabb[octadr, 0]                                   <L 120>
                var_53 = wp::address(var_oct_aabb, var_50, var_52);
                var_55 = wp::load(var_53);
                var_54 = wp::copy(var_55);
                var_11.center = var_54;
                // volume_data.half_size = oct_aabb[octadr, 1]                                <L 121>
                var_57 = wp::address(var_oct_aabb, var_50, var_56);
                var_59 = wp::load(var_57);
                var_58 = wp::copy(var_59);
                var_11.half_size = var_58;
                // volume_data.root = octadr                                                  <L 122>
                var_11.root = var_50;
                // volume_data.oct_aabb = oct_aabb                                            <L 123>
                var_11.oct_aabb = var_oct_aabb;
                // volume_data.oct_child = oct_child                                          <L 124>
                var_11.oct_child = var_oct_child;
                // volume_data.oct_coeff = oct_coeff                                          <L 125>
                var_11.oct_coeff = var_oct_coeff;
                // volume_data.valid = True                                                   <L 126>
                var_11.valid = var_60;
            }
        }
        var_61 = wp::where(var_23, var_29, var_50);
    }
    var_62 = wp::where(var_12, var_18, var_0);
    var_63 = wp::where(var_12, var_21, var_10);
    // return attributes, plugin_index, volume_data, MeshData()                               <L 128>
    var_64 = MeshData_52eaa0fa();
    ret_0 = var_62;
    ret_1 = var_63;
    ret_2 = var_11;
    ret_3 = var_64;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:158
static CUDA_CALLABLE wp::float32 sphere_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    wp::float32 var_3;
    //---------
    // forward
    // def sphere(p: wp.vec3, size: wp.vec3) -> float:                                        <L 159>
    // return wp.length(p) - size[0]                                                          <L 160>
    var_0 = wp::length(var_p);
    var_2 = wp::extract(var_size, var_1);
    var_3 = wp::sub(var_0, var_2);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:187
static CUDA_CALLABLE wp::float32 capsule_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def capsule(p: wp.vec3, size: wp.vec3) -> float:                                       <L 188>
    // r = size[0]                                                                            <L 189>
    var_1 = wp::extract(var_size, var_0);
    // h = size[1]                                                                            <L 190>
    var_3 = wp::extract(var_size, var_2);
    // pz_clamped = wp.clamp(p[2], -h, h)                                                     <L 191>
    var_5 = wp::extract(var_p, var_4);
    var_6 = wp::neg(var_3);
    var_7 = wp::clamp(var_5, var_6, var_3);
    // diff = wp.vec3(p[0], p[1], p[2] - pz_clamped)                                          <L 192>
    var_9 = wp::extract(var_p, var_8);
    var_11 = wp::extract(var_p, var_10);
    var_13 = wp::extract(var_p, var_12);
    var_14 = wp::sub(var_13, var_7);
    var_15 = wp::vec_t<3, wp::float32>(var_9, var_11, var_14);
    // return wp.length(diff) - r                                                             <L 193>
    var_16 = wp::length(var_15);
    var_17 = wp::sub(var_16, var_1);
    return var_17;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:196
static CUDA_CALLABLE wp::float32 cylinder_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    wp::vec_t<2, wp::float32> var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.0;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.0;
    wp::float32 var_19;
    const wp::float32 var_20 = 0.0;
    wp::float32 var_21;
    wp::vec_t<2, wp::float32> var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    //---------
    // forward
    // def cylinder(p: wp.vec3, size: wp.vec3) -> float:                                      <L 197>
    // r = size[0]                                                                            <L 198>
    var_1 = wp::extract(var_size, var_0);
    // h = size[1]                                                                            <L 199>
    var_3 = wp::extract(var_size, var_2);
    // dx = wp.length(wp.vec2(p[0], p[1])) - r                                                <L 200>
    var_5 = wp::extract(var_p, var_4);
    var_7 = wp::extract(var_p, var_6);
    var_8 = wp::vec_t<2, wp::float32>(var_5, var_7);
    var_9 = wp::length(var_8);
    var_10 = wp::sub(var_9, var_1);
    // dy = wp.abs(p[2]) - h                                                                  <L 201>
    var_12 = wp::extract(var_p, var_11);
    var_13 = wp::abs(var_12);
    var_14 = wp::sub(var_13, var_3);
    // return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))       <L 202>
    var_15 = wp::max(var_10, var_14);
    var_17 = wp::min(var_15, var_16);
    var_19 = wp::max(var_10, var_18);
    var_21 = wp::max(var_14, var_20);
    var_22 = wp::vec_t<2, wp::float32>(var_19, var_21);
    var_23 = wp::length(var_22);
    var_24 = wp::add(var_17, var_23);
    return var_24;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<3, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    return var_0;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:148
static CUDA_CALLABLE wp::vec_t<3, wp::float32> radial_field_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_size)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 2;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::float32 var_20;
    //---------
    // forward
    // def radial_field(a: wp.vec3, x: wp.vec3, size: wp.vec3) -> wp.vec3:                    <L 149>
    // field = wp.cw_div(-size, a)                                                            <L 150>
    var_0 = wp::neg(var_size);
    var_1 = wp::cw_div(var_0, var_a);
    // field = wp.normalize(field)                                                            <L 151>
    var_2 = wp::normalize(var_1);
    // field[0] *= wp.sign(x[0])                                                              <L 152>
    var_4 = wp::extract(var_x, var_3);
    var_5 = wp::sign(var_4);
    var_7 = wp::extract(var_2, var_6);
    var_8 = wp::mul(var_7, var_5);
    wp::assign_inplace(var_2, var_6, var_8);
    // field[1] *= wp.sign(x[1])                                                              <L 153>
    var_10 = wp::extract(var_x, var_9);
    var_11 = wp::sign(var_10);
    var_13 = wp::extract(var_2, var_12);
    var_14 = wp::mul(var_13, var_11);
    wp::assign_inplace(var_2, var_12, var_14);
    // field[2] *= wp.sign(x[2])                                                              <L 154>
    var_16 = wp::extract(var_x, var_15);
    var_17 = wp::sign(var_16);
    var_19 = wp::extract(var_2, var_18);
    var_20 = wp::mul(var_19, var_17);
    wp::assign_inplace(var_2, var_18, var_20);
    // return field                                                                           <L 155>
    return var_2;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:163
static CUDA_CALLABLE wp::float32 box_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    bool var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    bool var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    bool var_14;
    const wp::float32 var_15 = 0.0;
    const wp::float32 var_16 = 0.0;
    const wp::float32 var_17 = 0.0;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::float32 var_22 = 0.0;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    //---------
    // forward
    // def box(p: wp.vec3, size: wp.vec3) -> float:                                           <L 164>
    // a = wp.abs(p) - size                                                                   <L 165>
    var_0 = wp::abs(var_p);
    var_1 = wp::sub(var_0, var_size);
    // if a[0] >= 0 or a[1] >= 0 or a[2] >= 0:                                                <L 166>
    var_4 = wp::extract(var_1, var_3);
    var_6 = (var_4 >= var_5);
    var_2 = var_6;
    if (!var_2) {
        var_8 = wp::extract(var_1, var_7);
        var_10 = (var_8 >= var_9);
        var_2 = var_2 || var_10;
    }
    if (!var_2) {
        var_12 = wp::extract(var_1, var_11);
        var_14 = (var_12 >= var_13);
        var_2 = var_2 || var_14;
    }
    if (var_2) {
        // z = wp.vec3(0.0, 0.0, 0.0)                                                         <L 167>
        var_18 = wp::vec_t<3, wp::float32>(var_15, var_16, var_17);
        // b = wp.max(a, z)                                                                   <L 168>
        var_19 = wp::max(var_1, var_18);
        // return wp.norm_l2(b) + wp.min(wp.max(a), 0.0)                                      <L 169>
        var_20 = norm_l2_0(var_19);
        var_21 = wp::max(var_1);
        var_23 = wp::min(var_21, var_22);
        var_24 = wp::add(var_20, var_23);
        return var_24;
    }
    // b = radial_field(a, p, size)                                                           <L 170>
    var_25 = radial_field_0(var_1, var_p, var_size);
    // t = -wp.cw_div(a, wp.abs(b))                                                           <L 171>
    var_26 = wp::abs(var_25);
    var_27 = wp::cw_div(var_1, var_26);
    var_28 = wp::neg(var_27);
    // return -wp.min(t) * wp.norm_l2(b)                                                      <L 172>
    var_29 = wp::min(var_28);
    var_30 = wp::neg(var_29);
    var_31 = norm_l2_0(var_25);
    var_32 = wp::mul(var_30, var_31);
    return var_32;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:175
static CUDA_CALLABLE wp::float32 ellipsoid_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size)
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
    const wp::int32 var_10 = 2;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 0;
    wp::float32 var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    const wp::float32 var_21 = 2.0;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    const wp::float32 var_28 = 2.0;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 2;
    wp::float32 var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    const wp::float32 var_35 = 2.0;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    bool var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1e-12;
    wp::float32 var_44;
    const wp::float32 var_45 = 1.0;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    //---------
    // forward
    // def ellipsoid(p: wp.vec3, size: wp.vec3) -> float:                                     <L 176>
    // scaled_p = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])                     <L 177>
    var_1 = wp::extract(var_p, var_0);
    var_3 = wp::extract(var_size, var_2);
    var_4 = wp::div(var_1, var_3);
    var_6 = wp::extract(var_p, var_5);
    var_8 = wp::extract(var_size, var_7);
    var_9 = wp::div(var_6, var_8);
    var_11 = wp::extract(var_p, var_10);
    var_13 = wp::extract(var_size, var_12);
    var_14 = wp::div(var_11, var_13);
    var_15 = wp::vec_t<3, wp::float32>(var_4, var_9, var_14);
    // k0 = wp.length(scaled_p)                                                               <L 178>
    var_16 = wp::length(var_15);
    // k1 = wp.length(wp.vec3(p[0] / (size[0] ** 2.0), p[1] / (size[1] ** 2.0), p[2] / (size[2] ** 2.0)))       <L 179>
    var_18 = wp::extract(var_p, var_17);
    var_20 = wp::extract(var_size, var_19);
    var_22 = wp::pow(var_20, var_21);
    var_23 = wp::div(var_18, var_22);
    var_25 = wp::extract(var_p, var_24);
    var_27 = wp::extract(var_size, var_26);
    var_29 = wp::pow(var_27, var_28);
    var_30 = wp::div(var_25, var_29);
    var_32 = wp::extract(var_p, var_31);
    var_34 = wp::extract(var_size, var_33);
    var_36 = wp::pow(var_34, var_35);
    var_37 = wp::div(var_32, var_36);
    var_38 = wp::vec_t<3, wp::float32>(var_23, var_30, var_37);
    var_39 = wp::length(var_38);
    // if k1 != 0.0:                                                                          <L 180>
    var_41 = (var_39 != var_40);
    if (var_41) {
        // denom = k1                                                                         <L 181>
        var_42 = wp::copy(var_39);
    }
    if (!var_41) {
        // denom = 1e-12                                                                      <L 183>
    }
    var_44 = wp::where(var_41, var_42, var_43);
    // return k0 * (k0 - 1.0) / denom                                                         <L 184>
    var_46 = wp::sub(var_16, var_45);
    var_47 = wp::mul(var_16, var_46);
    var_48 = wp::div(var_47, var_44);
    return var_48;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:128
static CUDA_CALLABLE void _orthogonal_basis_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    const wp::float32 var_4 = 1.0;
    const wp::float32 var_5 = -1.0;
    wp::float32 var_6;
    const wp::float32 var_7 = -1.0;
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
    wp::float32 var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 0;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    const wp::int32 var_33 = 1;
    wp::float32 var_34;
    const wp::int32 var_35 = 1;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    //---------
    // forward
    // def _orthogonal_basis(vec: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:                        <L 129>
    // sign = wp.where(vec[2] >= 0.0, 1.0, -1.0)                                              <L 137>
    var_1 = wp::extract(var_vec, var_0);
    var_3 = (var_1 >= var_2);
    var_6 = wp::where(var_3, var_4, var_5);
    // a = -1.0 / (sign + vec[2])                                                             <L 138>
    var_9 = wp::extract(var_vec, var_8);
    var_10 = wp::add(var_6, var_9);
    var_11 = wp::div(var_7, var_10);
    // b = vec[0] * vec[1] * a                                                                <L 139>
    var_13 = wp::extract(var_vec, var_12);
    var_15 = wp::extract(var_vec, var_14);
    var_16 = wp::mul(var_13, var_15);
    var_17 = wp::mul(var_16, var_11);
    // b0 = wp.vec3(                                                                          <L 141>
    // 1.0 + sign * vec[0] * vec[0] * a,                                                      <L 142>
    var_20 = wp::extract(var_vec, var_19);
    var_21 = wp::mul(var_6, var_20);
    var_23 = wp::extract(var_vec, var_22);
    var_24 = wp::mul(var_21, var_23);
    var_25 = wp::mul(var_24, var_11);
    var_26 = wp::add(var_18, var_25);
    // sign * b,                                                                              <L 143>
    var_27 = wp::mul(var_6, var_17);
    // -sign * vec[0],                                                                        <L 144>
    var_28 = wp::neg(var_6);
    var_30 = wp::extract(var_vec, var_29);
    var_31 = wp::mul(var_28, var_30);
    var_32 = wp::vec_t<3, wp::float32>(var_26, var_27, var_31);
    // b1 = wp.vec3(                                                                          <L 146>
    // b,                                                                                     <L 147>
    // sign + vec[1] * vec[1] * a,                                                            <L 148>
    var_34 = wp::extract(var_vec, var_33);
    var_36 = wp::extract(var_vec, var_35);
    var_37 = wp::mul(var_34, var_36);
    var_38 = wp::mul(var_37, var_11);
    var_39 = wp::add(var_6, var_38);
    // -vec[1],                                                                               <L 149>
    var_41 = wp::extract(var_vec, var_40);
    var_42 = wp::neg(var_41);
    var_43 = wp::vec_t<3, wp::float32>(var_17, var_39, var_42);
    // return b0, b1                                                                          <L 151>
    ret_0 = var_32;
    ret_1 = var_43;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:154
static CUDA_CALLABLE void _ray_triangle_1(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> var_b0,
    wp::vec_t<3, wp::float32> var_b1,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    bool var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    const wp::float32 var_15 = 0.0;
    bool var_16;
    bool var_17;
    const wp::float32 var_18 = 0.0;
    bool var_19;
    const wp::float32 var_20 = 0.0;
    bool var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    bool var_24;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    const wp::float32 var_27 = 0.0;
    bool var_28;
    const wp::float32 var_29 = 0.0;
    bool var_30;
    bool var_31;
    const wp::float32 var_32 = 0.0;
    bool var_33;
    const wp::float32 var_34 = 0.0;
    bool var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    const wp::float32 var_38 = -1.0;
    wp::vec_t<3, wp::float32> var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::vec_t<2, wp::float32> var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::float32 var_51 = 1e-15;
    bool var_52;
    const wp::float32 var_53 = -1.0;
    wp::vec_t<3, wp::float32> var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    const wp::int32 var_64 = 0;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::int32 var_67 = 1;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    bool var_72;
    const wp::float32 var_73 = 0.0;
    bool var_74;
    const wp::float32 var_75 = 0.0;
    bool var_76;
    wp::float32 var_77;
    const wp::float32 var_78 = 1.0;
    bool var_79;
    const wp::float32 var_80 = -1.0;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::vec_t<3, wp::float32> var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    bool var_88;
    const wp::float32 var_89 = -1.0;
    wp::vec_t<3, wp::float32> var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    const wp::float32 var_94 = 0.0;
    bool var_95;
    const wp::float32 var_96 = -1.0;
    wp::float32 var_97;
    wp::vec_t<3, wp::float32> var_98;
    //---------
    // forward
    // def _ray_triangle(                                                                     <L 155>
    // dif0 = v0 - pnt                                                                        <L 159>
    var_0 = wp::sub(var_v0, var_pnt);
    // dif1 = v1 - pnt                                                                        <L 160>
    var_1 = wp::sub(var_v1, var_pnt);
    // dif2 = v2 - pnt                                                                        <L 161>
    var_2 = wp::sub(var_v2, var_pnt);
    // planar_00 = wp.dot(dif0, b0)                                                           <L 164>
    var_3 = wp::dot(var_0, var_b0);
    // planar_01 = wp.dot(dif0, b1)                                                           <L 165>
    var_4 = wp::dot(var_0, var_b1);
    // planar_10 = wp.dot(dif1, b0)                                                           <L 166>
    var_5 = wp::dot(var_1, var_b0);
    // planar_11 = wp.dot(dif1, b1)                                                           <L 167>
    var_6 = wp::dot(var_1, var_b1);
    // planar_20 = wp.dot(dif2, b0)                                                           <L 168>
    var_7 = wp::dot(var_2, var_b0);
    // planar_21 = wp.dot(dif2, b1)                                                           <L 169>
    var_8 = wp::dot(var_2, var_b1);
    // if (                                                                                   <L 172>
    // (planar_00 > 0.0 and planar_10 > 0.0 and planar_20 > 0.0)                              <L 173>
    var_12 = (var_3 > var_11);
    var_10 = var_12;
    if (var_10) {
        var_14 = (var_5 > var_13);
        var_10 = var_10 && var_14;
    }
    if (var_10) {
        var_16 = (var_7 > var_15);
        var_10 = var_10 && var_16;
    }
    var_9 = var_10;
    if (!var_9) {
        // or (planar_00 < 0.0 and planar_10 < 0.0 and planar_20 < 0.0)                       <L 174>
        var_19 = (var_3 < var_18);
        var_17 = var_19;
        if (var_17) {
            var_21 = (var_5 < var_20);
            var_17 = var_17 && var_21;
        }
        if (var_17) {
            var_23 = (var_7 < var_22);
            var_17 = var_17 && var_23;
        }
        var_9 = var_9 || var_17;
    }
    if (!var_9) {
        // or (planar_01 > 0.0 and planar_11 > 0.0 and planar_21 > 0.0)                       <L 175>
        var_26 = (var_4 > var_25);
        var_24 = var_26;
        if (var_24) {
            var_28 = (var_6 > var_27);
            var_24 = var_24 && var_28;
        }
        if (var_24) {
            var_30 = (var_8 > var_29);
            var_24 = var_24 && var_30;
        }
        var_9 = var_9 || var_24;
    }
    if (!var_9) {
        // or (planar_01 < 0.0 and planar_11 < 0.0 and planar_21 < 0.0)                       <L 176>
        var_33 = (var_4 < var_32);
        var_31 = var_33;
        if (var_31) {
            var_35 = (var_6 < var_34);
            var_31 = var_31 && var_35;
        }
        if (var_31) {
            var_37 = (var_8 < var_36);
            var_31 = var_31 && var_37;
        }
        var_9 = var_9 || var_31;
    }
    if (var_9) {
        // return -1.0, wp.vec3()                                                             <L 178>
        var_39 = wp::vec_t<3, wp::float32>();
        ret_0 = var_38;
        ret_1 = var_39;
        return;
    }
    // A00 = planar_00 - planar_20                                                            <L 182>
    var_40 = wp::sub(var_3, var_7);
    // A10 = planar_10 - planar_20                                                            <L 183>
    var_41 = wp::sub(var_5, var_7);
    // A01 = planar_01 - planar_21                                                            <L 184>
    var_42 = wp::sub(var_4, var_8);
    // A11 = planar_11 - planar_21                                                            <L 185>
    var_43 = wp::sub(var_6, var_8);
    // b = wp.vec2(-planar_20, -planar_21)                                                    <L 187>
    var_44 = wp::neg(var_7);
    var_45 = wp::neg(var_8);
    var_46 = wp::vec_t<2, wp::float32>(var_44, var_45);
    // det = A00 * A11 - A10 * A01                                                            <L 189>
    var_47 = wp::mul(var_40, var_43);
    var_48 = wp::mul(var_41, var_42);
    var_49 = wp::sub(var_47, var_48);
    // if wp.abs(det) < MJ_MINVAL:                                                            <L 190>
    var_50 = wp::abs(var_49);
    var_52 = (var_50 < var_51);
    if (var_52) {
        // return -1.0, wp.vec3()                                                             <L 191>
        var_54 = wp::vec_t<3, wp::float32>();
        ret_0 = var_53;
        ret_1 = var_54;
        return;
    }
    // t0 = (A11 * b[0] - A10 * b[1]) / det                                                   <L 193>
    var_56 = wp::extract(var_46, var_55);
    var_57 = wp::mul(var_43, var_56);
    var_59 = wp::extract(var_46, var_58);
    var_60 = wp::mul(var_41, var_59);
    var_61 = wp::sub(var_57, var_60);
    var_62 = wp::div(var_61, var_49);
    // t1 = (-A01 * b[0] + A00 * b[1]) / det                                                  <L 194>
    var_63 = wp::neg(var_42);
    var_65 = wp::extract(var_46, var_64);
    var_66 = wp::mul(var_63, var_65);
    var_68 = wp::extract(var_46, var_67);
    var_69 = wp::mul(var_40, var_68);
    var_70 = wp::add(var_66, var_69);
    var_71 = wp::div(var_70, var_49);
    // if t0 < 0.0 or t1 < 0.0 or t0 + t1 > 1.0:                                              <L 197>
    var_74 = (var_62 < var_73);
    var_72 = var_74;
    if (!var_72) {
        var_76 = (var_71 < var_75);
        var_72 = var_72 || var_76;
    }
    if (!var_72) {
        var_77 = wp::add(var_62, var_71);
        var_79 = (var_77 > var_78);
        var_72 = var_72 || var_79;
    }
    if (var_72) {
        // return -1.0, wp.vec3()                                                             <L 198>
        var_81 = wp::vec_t<3, wp::float32>();
        ret_0 = var_80;
        ret_1 = var_81;
        return;
    }
    // dif0 = v0 - v2                                                                         <L 201>
    var_82 = wp::sub(var_v0, var_v2);
    // dif1 = v1 - v2                                                                         <L 202>
    var_83 = wp::sub(var_v1, var_v2);
    // dif2 = pnt - v2                                                                        <L 203>
    var_84 = wp::sub(var_pnt, var_v2);
    // nrm = wp.cross(dif0, dif1)  # normal to triangle plane                                 <L 204>
    var_85 = wp::cross(var_82, var_83);
    // denom = wp.dot(vec, nrm)                                                               <L 205>
    var_86 = wp::dot(var_vec, var_85);
    // if wp.abs(denom) < MJ_MINVAL:                                                          <L 206>
    var_87 = wp::abs(var_86);
    var_88 = (var_87 < var_51);
    if (var_88) {
        // return -1.0, wp.vec3()                                                             <L 207>
        var_90 = wp::vec_t<3, wp::float32>();
        ret_0 = var_89;
        ret_1 = var_90;
        return;
    }
    // dist = -wp.dot(dif2, nrm) / denom                                                      <L 209>
    var_91 = wp::dot(var_84, var_85);
    var_92 = wp::neg(var_91);
    var_93 = wp::div(var_92, var_86);
    // return wp.where(dist >= 0.0, dist, -1.0), wp.normalize(nrm)                            <L 210>
    var_95 = (var_93 >= var_94);
    var_97 = wp::where(var_95, var_93, var_96);
    var_98 = wp::normalize(var_85);
    ret_0 = var_97;
    ret_1 = var_98;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:628
static CUDA_CALLABLE void ray_mesh_0(
    wp::int32 var_nmeshface,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_faceadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::vec_t<3, wp::int32>> var_mesh_face,
    wp::int32 var_data_id,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<6, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::float32 var_3 = 0.0;
    bool var_4;
    const wp::float32 var_5 = -1.0;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    const wp::float32 var_11 = -1.0;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::int32* var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32* var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 1;
    wp::int32 var_21;
    wp::shape_t* var_22;
    const wp::int32 var_23 = 0;
    wp::int32 var_24;
    wp::shape_t var_25;
    bool var_26;
    const wp::int32 var_27 = 1;
    wp::int32 var_28;
    wp::int32* var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    wp::int32 var_32;
    wp::int32 var_33;
    wp::range_t var_34;
    wp::int32 var_35;
    wp::vec_t<3, wp::int32>* var_36;
    wp::vec_t<3, wp::int32> var_37;
    wp::vec_t<3, wp::int32> var_38;
    const wp::int32 var_39 = 0;
    wp::int32 var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    const wp::int32 var_45 = 1;
    wp::int32 var_46;
    wp::int32 var_47;
    wp::vec_t<3, wp::float32>* var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    const wp::int32 var_51 = 2;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::vec_t<3, wp::float32>* var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    wp::vec_t<3, wp::float32> var_58;
    bool var_59;
    const wp::int32 var_60 = 0;
    bool var_61;
    bool var_62;
    const wp::int32 var_63 = 0;
    bool var_64;
    bool var_65;
    wp::float32 var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::float32 var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    //---------
    // forward
    // def ray_mesh(                                                                          <L 629>
    // dist_box, _all, _normal = ray_box(pos, mat, size, pnt, vec)                            <L 646>
    ray_box_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_0, var_1, var_2);
    // if dist_box < 0.0:                                                                     <L 647>
    var_4 = (var_0 < var_3);
    if (var_4) {
        // return -1.0, wp.vec3()                                                             <L 648>
        var_6 = wp::vec_t<3, wp::float32>();
        ret_0 = var_5;
        ret_1 = var_6;
        return;
    }
    // pnt, vec = _ray_map(pos, mat, pnt, vec)                                                <L 650>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_7, var_8);
    // b0, b1 = _orthogonal_basis(vec)                                                        <L 653>
    _orthogonal_basis_0(var_8, var_9, var_10);
    // x = float(-1.0)                                                                        <L 655>
    var_12 = wp::float(var_11);
    // normal = wp.vec3()                                                                     <L 656>
    var_13 = wp::vec_t<3, wp::float32>();
    // vert_start = mesh_vertadr[data_id]                                                     <L 659>
    var_14 = wp::address(var_mesh_vertadr, var_data_id);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // face_start = mesh_faceadr[data_id]                                                     <L 662>
    var_17 = wp::address(var_mesh_faceadr, var_data_id);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // if data_id + 1 < mesh_faceadr.shape[0]:                                                <L 664>
    var_21 = wp::add(var_data_id, var_20);
    var_22 = &(var_mesh_faceadr.shape);
    var_25 = wp::load(var_22);
    var_24 = wp::extract(var_25, var_23);
    var_26 = (var_21 < var_24);
    if (var_26) {
        // face_end = mesh_faceadr[data_id + 1]                                               <L 665>
        var_28 = wp::add(var_data_id, var_27);
        var_29 = wp::address(var_mesh_faceadr, var_28);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
    }
    if (!var_26) {
        // face_end = nmeshface                                                               <L 667>
        var_32 = wp::copy(var_nmeshface);
    }
    var_33 = wp::where(var_26, var_30, var_32);
    // for i in range(face_start, face_end):                                                  <L 670>
    var_34 = wp::range(var_18, var_33);
    start_for_1:;
        if (iter_cmp(var_34) == 0) goto end_for_1;
        var_35 = wp::iter_next(var_34);
        // v_idx = mesh_face[i]                                                               <L 672>
        var_36 = wp::address(var_mesh_face, var_35);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // v0 = mesh_vert[vert_start + v_idx.x]                                               <L 675>
        var_40 = wp::extract(var_37, var_39);
        var_41 = wp::add(var_15, var_40);
        var_42 = wp::address(var_mesh_vert, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // v1 = mesh_vert[vert_start + v_idx.y]                                               <L 676>
        var_46 = wp::extract(var_37, var_45);
        var_47 = wp::add(var_15, var_46);
        var_48 = wp::address(var_mesh_vert, var_47);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // v2 = mesh_vert[vert_start + v_idx.z]                                               <L 677>
        var_52 = wp::extract(var_37, var_51);
        var_53 = wp::add(var_15, var_52);
        var_54 = wp::address(var_mesh_vert, var_53);
        var_56 = wp::load(var_54);
        var_55 = wp::copy(var_56);
        // dist, normal_tri = _ray_triangle(v0, v1, v2, pnt, vec, b0, b1)                     <L 680>
        _ray_triangle_1(var_43, var_49, var_55, var_7, var_8, var_9, var_10, var_57, var_58);
        // if dist >= 0 and (x < 0 or dist < x):                                              <L 681>
        var_61 = (var_57 >= var_60);
        var_59 = var_61;
        if (var_59) {
            var_64 = (var_12 < var_63);
            var_62 = var_64;
            if (!var_62) {
                var_65 = (var_57 < var_12);
                var_62 = var_62 || var_65;
            }
            var_59 = var_59 && var_62;
        }
        if (var_59) {
            // x = dist                                                                       <L 682>
            var_66 = wp::copy(var_57);
            // normal = normal_tri                                                            <L 683>
            var_67 = wp::copy(var_58);
        }
        var_68 = wp::where(var_59, var_66, var_12);
        var_69 = wp::where(var_59, var_67, var_13);
        wp::assign(var_12, var_68);
        wp::assign(var_13, var_69);
        goto start_for_1;
    end_for_1:;
    // normal = mat @ normal                                                                  <L 685>
    var_70 = wp::mul(var_mat, var_13);
    // return x, normal                                                                       <L 687>
    ret_0 = var_12;
    ret_1 = var_70;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:391
static CUDA_CALLABLE void box_project_0(
    wp::vec_t<3, wp::float32> var_center,
    wp::vec_t<3, wp::float32> var_half_size,
    wp::vec_t<3, wp::float32> var_xyz,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 2;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    bool var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    const wp::float32 var_23 = 0.0;
    bool var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    const wp::float32 var_27 = 0.0;
    bool var_28;
    const wp::int32 var_29 = 2;
    wp::float32 var_30;
    const wp::float32 var_31 = 0.0;
    bool var_32;
    const wp::float32 var_33 = 0.0;
    const wp::float32 var_34 = 0.0;
    const wp::float32 var_35 = 0.0001;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    const wp::int32 var_47 = 0;
    wp::float32 var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 0;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 1;
    wp::float32 var_64;
    const wp::int32 var_65 = 2;
    wp::float32 var_66;
    wp::vec_t<3, wp::float32> var_67;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    const wp::int32 var_70 = 0;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::float32 var_80;
    wp::vec_t<3, wp::float32> var_81;
    const wp::int32 var_82 = 1;
    wp::float32 var_83;
    const wp::float32 var_84 = 0.0;
    bool var_85;
    const wp::int32 var_86 = 1;
    wp::float32 var_87;
    const wp::int32 var_88 = 1;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 1;
    wp::float32 var_93;
    const wp::float32 var_94 = 0.0;
    bool var_95;
    const wp::int32 var_96 = 0;
    wp::float32 var_97;
    const wp::int32 var_98 = 1;
    wp::float32 var_99;
    const wp::int32 var_100 = 1;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    const wp::int32 var_104 = 2;
    wp::float32 var_105;
    wp::vec_t<3, wp::float32> var_106;
    const wp::int32 var_107 = 0;
    wp::float32 var_108;
    const wp::int32 var_109 = 1;
    wp::float32 var_110;
    const wp::int32 var_111 = 1;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 2;
    wp::float32 var_116;
    wp::vec_t<3, wp::float32> var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::float32 var_119;
    wp::vec_t<3, wp::float32> var_120;
    const wp::int32 var_121 = 2;
    wp::float32 var_122;
    const wp::float32 var_123 = 0.0;
    bool var_124;
    const wp::int32 var_125 = 2;
    wp::float32 var_126;
    const wp::int32 var_127 = 2;
    wp::float32 var_128;
    wp::float32 var_129;
    wp::float32 var_130;
    const wp::int32 var_131 = 2;
    wp::float32 var_132;
    const wp::float32 var_133 = 0.0;
    bool var_134;
    const wp::int32 var_135 = 0;
    wp::float32 var_136;
    const wp::int32 var_137 = 1;
    wp::float32 var_138;
    const wp::int32 var_139 = 2;
    wp::float32 var_140;
    const wp::int32 var_141 = 2;
    wp::float32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    wp::vec_t<3, wp::float32> var_145;
    const wp::int32 var_146 = 0;
    wp::float32 var_147;
    const wp::int32 var_148 = 1;
    wp::float32 var_149;
    const wp::int32 var_150 = 2;
    wp::float32 var_151;
    const wp::int32 var_152 = 2;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::vec_t<3, wp::float32> var_157;
    wp::float32 var_158;
    wp::vec_t<3, wp::float32> var_159;
    wp::float32 var_160;
    //---------
    // forward
    // def box_project(center: wp.vec3, half_size: wp.vec3, xyz: wp.vec3) -> Tuple[float, wp.vec3]:       <L 392>
    // r = xyz - center                                                                       <L 393>
    var_0 = wp::sub(var_xyz, var_center);
    // q = wp.vec3(wp.abs(r[0]) - half_size[0], wp.abs(r[1]) - half_size[1], wp.abs(r[2]) - half_size[2])       <L 394>
    var_2 = wp::extract(var_0, var_1);
    var_3 = wp::abs(var_2);
    var_5 = wp::extract(var_half_size, var_4);
    var_6 = wp::sub(var_3, var_5);
    var_8 = wp::extract(var_0, var_7);
    var_9 = wp::abs(var_8);
    var_11 = wp::extract(var_half_size, var_10);
    var_12 = wp::sub(var_9, var_11);
    var_14 = wp::extract(var_0, var_13);
    var_15 = wp::abs(var_14);
    var_17 = wp::extract(var_half_size, var_16);
    var_18 = wp::sub(var_15, var_17);
    var_19 = wp::vec_t<3, wp::float32>(var_6, var_12, var_18);
    // if q[0] <= 0.0 and q[1] <= 0.0 and q[2] <= 0.0:                                        <L 396>
    var_22 = wp::extract(var_19, var_21);
    var_24 = (var_22 <= var_23);
    var_20 = var_24;
    if (var_20) {
        var_26 = wp::extract(var_19, var_25);
        var_28 = (var_26 <= var_27);
        var_20 = var_20 && var_28;
    }
    if (var_20) {
        var_30 = wp::extract(var_19, var_29);
        var_32 = (var_30 <= var_31);
        var_20 = var_20 && var_32;
    }
    if (var_20) {
        // return 0.0, xyz                                                                    <L 397>
        ret_0 = var_33;
        ret_1 = var_xyz;
        return;
    }
    if (!var_20) {
        // dist_sqr = 0.0                                                                     <L 400>
        // eps = 1e-4                                                                         <L 401>
        // point = wp.vec3(xyz[0], xyz[1], xyz[2])                                            <L 402>
        var_37 = wp::extract(var_xyz, var_36);
        var_39 = wp::extract(var_xyz, var_38);
        var_41 = wp::extract(var_xyz, var_40);
        var_42 = wp::vec_t<3, wp::float32>(var_37, var_39, var_41);
        // if q[0] >= 0.0:                                                                    <L 404>
        var_44 = wp::extract(var_19, var_43);
        var_46 = (var_44 >= var_45);
        if (var_46) {
            // dist_sqr += q[0] * q[0]                                                        <L 405>
            var_48 = wp::extract(var_19, var_47);
            var_50 = wp::extract(var_19, var_49);
            var_51 = wp::mul(var_48, var_50);
            var_52 = wp::add(var_34, var_51);
            // if r[0] > 0.0:                                                                 <L 406>
            var_54 = wp::extract(var_0, var_53);
            var_56 = (var_54 > var_55);
            if (var_56) {
                // point = wp.vec3(point[0] - (q[0] + eps), point[1], point[2])               <L 407>
                var_58 = wp::extract(var_42, var_57);
                var_60 = wp::extract(var_19, var_59);
                var_61 = wp::add(var_60, var_35);
                var_62 = wp::sub(var_58, var_61);
                var_64 = wp::extract(var_42, var_63);
                var_66 = wp::extract(var_42, var_65);
                var_67 = wp::vec_t<3, wp::float32>(var_62, var_64, var_66);
            }
            if (!var_56) {
                // point = wp.vec3(point[0] + (q[0] + eps), point[1], point[2])               <L 409>
                var_69 = wp::extract(var_42, var_68);
                var_71 = wp::extract(var_19, var_70);
                var_72 = wp::add(var_71, var_35);
                var_73 = wp::add(var_69, var_72);
                var_75 = wp::extract(var_42, var_74);
                var_77 = wp::extract(var_42, var_76);
                var_78 = wp::vec_t<3, wp::float32>(var_73, var_75, var_77);
            }
            var_79 = wp::where(var_56, var_67, var_78);
        }
        var_80 = wp::where(var_46, var_52, var_34);
        var_81 = wp::where(var_46, var_79, var_42);
        // if q[1] >= 0.0:                                                                    <L 411>
        var_83 = wp::extract(var_19, var_82);
        var_85 = (var_83 >= var_84);
        if (var_85) {
            // dist_sqr += q[1] * q[1]                                                        <L 412>
            var_87 = wp::extract(var_19, var_86);
            var_89 = wp::extract(var_19, var_88);
            var_90 = wp::mul(var_87, var_89);
            var_91 = wp::add(var_80, var_90);
            // if r[1] > 0.0:                                                                 <L 413>
            var_93 = wp::extract(var_0, var_92);
            var_95 = (var_93 > var_94);
            if (var_95) {
                // point = wp.vec3(point[0], point[1] - (q[1] + eps), point[2])               <L 414>
                var_97 = wp::extract(var_81, var_96);
                var_99 = wp::extract(var_81, var_98);
                var_101 = wp::extract(var_19, var_100);
                var_102 = wp::add(var_101, var_35);
                var_103 = wp::sub(var_99, var_102);
                var_105 = wp::extract(var_81, var_104);
                var_106 = wp::vec_t<3, wp::float32>(var_97, var_103, var_105);
            }
            if (!var_95) {
                // point = wp.vec3(point[0], point[1] + (q[1] + eps), point[2])               <L 416>
                var_108 = wp::extract(var_81, var_107);
                var_110 = wp::extract(var_81, var_109);
                var_112 = wp::extract(var_19, var_111);
                var_113 = wp::add(var_112, var_35);
                var_114 = wp::add(var_110, var_113);
                var_116 = wp::extract(var_81, var_115);
                var_117 = wp::vec_t<3, wp::float32>(var_108, var_114, var_116);
            }
            var_118 = wp::where(var_95, var_106, var_117);
        }
        var_119 = wp::where(var_85, var_91, var_80);
        var_120 = wp::where(var_85, var_118, var_81);
        // if q[2] >= 0.0:                                                                    <L 418>
        var_122 = wp::extract(var_19, var_121);
        var_124 = (var_122 >= var_123);
        if (var_124) {
            // dist_sqr += q[2] * q[2]                                                        <L 419>
            var_126 = wp::extract(var_19, var_125);
            var_128 = wp::extract(var_19, var_127);
            var_129 = wp::mul(var_126, var_128);
            var_130 = wp::add(var_119, var_129);
            // if r[2] > 0.0:                                                                 <L 420>
            var_132 = wp::extract(var_0, var_131);
            var_134 = (var_132 > var_133);
            if (var_134) {
                // point = wp.vec3(point[0], point[1], point[2] - (q[2] + eps))               <L 421>
                var_136 = wp::extract(var_120, var_135);
                var_138 = wp::extract(var_120, var_137);
                var_140 = wp::extract(var_120, var_139);
                var_142 = wp::extract(var_19, var_141);
                var_143 = wp::add(var_142, var_35);
                var_144 = wp::sub(var_140, var_143);
                var_145 = wp::vec_t<3, wp::float32>(var_136, var_138, var_144);
            }
            if (!var_134) {
                // point = wp.vec3(point[0], point[1], point[2] + (q[2] + eps))               <L 423>
                var_147 = wp::extract(var_120, var_146);
                var_149 = wp::extract(var_120, var_148);
                var_151 = wp::extract(var_120, var_150);
                var_153 = wp::extract(var_19, var_152);
                var_154 = wp::add(var_153, var_35);
                var_155 = wp::add(var_151, var_154);
                var_156 = wp::vec_t<3, wp::float32>(var_147, var_149, var_155);
            }
            var_157 = wp::where(var_134, var_145, var_156);
        }
        var_158 = wp::where(var_124, var_130, var_119);
        var_159 = wp::where(var_124, var_157, var_120);
        // return wp.sqrt(dist_sqr), point                                                    <L 425>
        var_160 = wp::sqrt(var_158);
        ret_0 = var_160;
        ret_1 = var_159;
        return;
    }
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:318
static CUDA_CALLABLE void find_oct_0(
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::vec_t<3, wp::float32> var_p,
    bool var_grad,
    wp::int32 var_root,
    wp::int32 & ret_0,
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> & ret_1)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 100;
    wp::int32 var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<8, wp::float32> var_4;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<8, wp::float32> var_6;
    const wp::float32 var_7 = 0.0;
    wp::vec_t<8, wp::float32> var_8;
    const wp::float32 var_9 = 1e-06;
    const wp::int32 var_10 = 0;
    bool var_11;
    const wp::int32 var_12 = 1;
    wp::int32 var_13;
    wp::int32 var_14;
    const wp::int32 var_15 = -1;
    bool var_16;
    const wp::str var_17 = "ERROR: Invalid node number\n";
    const wp::int32 var_18 = -1;
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> var_19;
    const wp::int32 var_20 = 0;
    wp::vec_t<3, wp::float32>* var_21;
    const wp::int32 var_22 = 1;
    wp::vec_t<3, wp::float32>* var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    const wp::int32 var_27 = 0;
    wp::vec_t<3, wp::float32>* var_28;
    const wp::int32 var_29 = 1;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    bool var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::int32 var_38 = 0;
    wp::float32 var_39;
    bool var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    bool var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::int32 var_50 = 1;
    wp::float32 var_51;
    bool var_52;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    bool var_58;
    const wp::int32 var_59 = 2;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    bool var_64;
    const wp::int32 var_65 = 2;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::int32 var_68 = 2;
    wp::float32 var_69;
    bool var_70;
    wp::int32 var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<8, wp::int32>* var_75;
    const wp::int32 var_76 = 0;
    wp::int32 var_77;
    wp::vec_t<8, wp::int32> var_78;
    const wp::int32 var_79 = -1;
    bool var_80;
    wp::int32 var_81;
    wp::vec_t<8, wp::int32>* var_82;
    const wp::int32 var_83 = 1;
    wp::int32 var_84;
    wp::vec_t<8, wp::int32> var_85;
    const wp::int32 var_86 = -1;
    bool var_87;
    wp::int32 var_88;
    wp::int32 var_89;
    wp::vec_t<8, wp::int32>* var_90;
    const wp::int32 var_91 = 2;
    wp::int32 var_92;
    wp::vec_t<8, wp::int32> var_93;
    const wp::int32 var_94 = -1;
    bool var_95;
    wp::int32 var_96;
    wp::int32 var_97;
    wp::vec_t<8, wp::int32>* var_98;
    const wp::int32 var_99 = 3;
    wp::int32 var_100;
    wp::vec_t<8, wp::int32> var_101;
    const wp::int32 var_102 = -1;
    bool var_103;
    wp::int32 var_104;
    wp::int32 var_105;
    wp::vec_t<8, wp::int32>* var_106;
    const wp::int32 var_107 = 4;
    wp::int32 var_108;
    wp::vec_t<8, wp::int32> var_109;
    const wp::int32 var_110 = -1;
    bool var_111;
    wp::int32 var_112;
    wp::int32 var_113;
    wp::vec_t<8, wp::int32>* var_114;
    const wp::int32 var_115 = 5;
    wp::int32 var_116;
    wp::vec_t<8, wp::int32> var_117;
    const wp::int32 var_118 = -1;
    bool var_119;
    wp::int32 var_120;
    wp::int32 var_121;
    wp::vec_t<8, wp::int32>* var_122;
    const wp::int32 var_123 = 6;
    wp::int32 var_124;
    wp::vec_t<8, wp::int32> var_125;
    const wp::int32 var_126 = -1;
    bool var_127;
    wp::int32 var_128;
    wp::int32 var_129;
    wp::vec_t<8, wp::int32>* var_130;
    const wp::int32 var_131 = 7;
    wp::int32 var_132;
    wp::vec_t<8, wp::int32> var_133;
    const wp::int32 var_134 = -1;
    bool var_135;
    wp::int32 var_136;
    wp::int32 var_137;
    const wp::int32 var_138 = 0;
    bool var_139;
    const wp::int32 var_140 = 0;
    bool var_141;
    const wp::int32 var_142 = 1;
    wp::int32 var_143;
    const wp::int32 var_144 = 0;
    wp::float32 var_145;
    const wp::float32 var_146 = 1.0;
    const wp::int32 var_147 = 0;
    wp::float32 var_148;
    wp::float32 var_149;
    wp::float32 var_150;
    const wp::int32 var_151 = 2;
    wp::int32 var_152;
    const wp::int32 var_153 = 1;
    wp::float32 var_154;
    const wp::float32 var_155 = 1.0;
    const wp::int32 var_156 = 1;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    const wp::int32 var_161 = 4;
    wp::int32 var_162;
    const wp::int32 var_163 = 2;
    wp::float32 var_164;
    const wp::float32 var_165 = 1.0;
    const wp::int32 var_166 = 2;
    wp::float32 var_167;
    wp::float32 var_168;
    wp::float32 var_169;
    wp::float32 var_170;
    const wp::int32 var_171 = 1;
    wp::int32 var_172;
    const wp::float32 var_173 = 1.0;
    const wp::float32 var_174 = -1.0;
    wp::float32 var_175;
    const wp::int32 var_176 = 2;
    wp::int32 var_177;
    const wp::int32 var_178 = 1;
    wp::float32 var_179;
    const wp::float32 var_180 = 1.0;
    const wp::int32 var_181 = 1;
    wp::float32 var_182;
    wp::float32 var_183;
    wp::float32 var_184;
    wp::float32 var_185;
    const wp::int32 var_186 = 4;
    wp::int32 var_187;
    const wp::int32 var_188 = 2;
    wp::float32 var_189;
    const wp::float32 var_190 = 1.0;
    const wp::int32 var_191 = 2;
    wp::float32 var_192;
    wp::float32 var_193;
    wp::float32 var_194;
    wp::float32 var_195;
    const wp::int32 var_196 = 1;
    wp::int32 var_197;
    const wp::int32 var_198 = 0;
    wp::float32 var_199;
    const wp::float32 var_200 = 1.0;
    const wp::int32 var_201 = 0;
    wp::float32 var_202;
    wp::float32 var_203;
    wp::float32 var_204;
    const wp::int32 var_205 = 2;
    wp::int32 var_206;
    const wp::float32 var_207 = 1.0;
    const wp::float32 var_208 = -1.0;
    wp::float32 var_209;
    wp::float32 var_210;
    const wp::int32 var_211 = 4;
    wp::int32 var_212;
    const wp::int32 var_213 = 2;
    wp::float32 var_214;
    const wp::float32 var_215 = 1.0;
    const wp::int32 var_216 = 2;
    wp::float32 var_217;
    wp::float32 var_218;
    wp::float32 var_219;
    wp::float32 var_220;
    const wp::int32 var_221 = 1;
    wp::int32 var_222;
    const wp::int32 var_223 = 0;
    wp::float32 var_224;
    const wp::float32 var_225 = 1.0;
    const wp::int32 var_226 = 0;
    wp::float32 var_227;
    wp::float32 var_228;
    wp::float32 var_229;
    const wp::int32 var_230 = 2;
    wp::int32 var_231;
    const wp::int32 var_232 = 1;
    wp::float32 var_233;
    const wp::float32 var_234 = 1.0;
    const wp::int32 var_235 = 1;
    wp::float32 var_236;
    wp::float32 var_237;
    wp::float32 var_238;
    wp::float32 var_239;
    const wp::int32 var_240 = 4;
    wp::int32 var_241;
    const wp::float32 var_242 = 1.0;
    const wp::float32 var_243 = -1.0;
    wp::float32 var_244;
    wp::float32 var_245;
    const wp::int32 var_246 = 1;
    bool var_247;
    const wp::int32 var_248 = 1;
    wp::int32 var_249;
    const wp::int32 var_250 = 0;
    wp::float32 var_251;
    const wp::float32 var_252 = 1.0;
    const wp::int32 var_253 = 0;
    wp::float32 var_254;
    wp::float32 var_255;
    wp::float32 var_256;
    const wp::int32 var_257 = 2;
    wp::int32 var_258;
    const wp::int32 var_259 = 1;
    wp::float32 var_260;
    const wp::float32 var_261 = 1.0;
    const wp::int32 var_262 = 1;
    wp::float32 var_263;
    wp::float32 var_264;
    wp::float32 var_265;
    wp::float32 var_266;
    const wp::int32 var_267 = 4;
    wp::int32 var_268;
    const wp::int32 var_269 = 2;
    wp::float32 var_270;
    const wp::float32 var_271 = 1.0;
    const wp::int32 var_272 = 2;
    wp::float32 var_273;
    wp::float32 var_274;
    wp::float32 var_275;
    wp::float32 var_276;
    const wp::int32 var_277 = 1;
    wp::int32 var_278;
    const wp::float32 var_279 = 1.0;
    const wp::float32 var_280 = -1.0;
    wp::float32 var_281;
    const wp::int32 var_282 = 2;
    wp::int32 var_283;
    const wp::int32 var_284 = 1;
    wp::float32 var_285;
    const wp::float32 var_286 = 1.0;
    const wp::int32 var_287 = 1;
    wp::float32 var_288;
    wp::float32 var_289;
    wp::float32 var_290;
    wp::float32 var_291;
    const wp::int32 var_292 = 4;
    wp::int32 var_293;
    const wp::int32 var_294 = 2;
    wp::float32 var_295;
    const wp::float32 var_296 = 1.0;
    const wp::int32 var_297 = 2;
    wp::float32 var_298;
    wp::float32 var_299;
    wp::float32 var_300;
    wp::float32 var_301;
    const wp::int32 var_302 = 1;
    wp::int32 var_303;
    const wp::int32 var_304 = 0;
    wp::float32 var_305;
    const wp::float32 var_306 = 1.0;
    const wp::int32 var_307 = 0;
    wp::float32 var_308;
    wp::float32 var_309;
    wp::float32 var_310;
    const wp::int32 var_311 = 2;
    wp::int32 var_312;
    const wp::float32 var_313 = 1.0;
    const wp::float32 var_314 = -1.0;
    wp::float32 var_315;
    wp::float32 var_316;
    const wp::int32 var_317 = 4;
    wp::int32 var_318;
    const wp::int32 var_319 = 2;
    wp::float32 var_320;
    const wp::float32 var_321 = 1.0;
    const wp::int32 var_322 = 2;
    wp::float32 var_323;
    wp::float32 var_324;
    wp::float32 var_325;
    wp::float32 var_326;
    const wp::int32 var_327 = 1;
    wp::int32 var_328;
    const wp::int32 var_329 = 0;
    wp::float32 var_330;
    const wp::float32 var_331 = 1.0;
    const wp::int32 var_332 = 0;
    wp::float32 var_333;
    wp::float32 var_334;
    wp::float32 var_335;
    const wp::int32 var_336 = 2;
    wp::int32 var_337;
    const wp::int32 var_338 = 1;
    wp::float32 var_339;
    const wp::float32 var_340 = 1.0;
    const wp::int32 var_341 = 1;
    wp::float32 var_342;
    wp::float32 var_343;
    wp::float32 var_344;
    wp::float32 var_345;
    const wp::int32 var_346 = 4;
    wp::int32 var_347;
    const wp::float32 var_348 = 1.0;
    const wp::float32 var_349 = -1.0;
    wp::float32 var_350;
    wp::float32 var_351;
    const wp::int32 var_352 = 2;
    bool var_353;
    const wp::int32 var_354 = 1;
    wp::int32 var_355;
    const wp::int32 var_356 = 0;
    wp::float32 var_357;
    const wp::float32 var_358 = 1.0;
    const wp::int32 var_359 = 0;
    wp::float32 var_360;
    wp::float32 var_361;
    wp::float32 var_362;
    const wp::int32 var_363 = 2;
    wp::int32 var_364;
    const wp::int32 var_365 = 1;
    wp::float32 var_366;
    const wp::float32 var_367 = 1.0;
    const wp::int32 var_368 = 1;
    wp::float32 var_369;
    wp::float32 var_370;
    wp::float32 var_371;
    wp::float32 var_372;
    const wp::int32 var_373 = 4;
    wp::int32 var_374;
    const wp::int32 var_375 = 2;
    wp::float32 var_376;
    const wp::float32 var_377 = 1.0;
    const wp::int32 var_378 = 2;
    wp::float32 var_379;
    wp::float32 var_380;
    wp::float32 var_381;
    wp::float32 var_382;
    const wp::int32 var_383 = 1;
    wp::int32 var_384;
    const wp::float32 var_385 = 1.0;
    const wp::float32 var_386 = -1.0;
    wp::float32 var_387;
    const wp::int32 var_388 = 2;
    wp::int32 var_389;
    const wp::int32 var_390 = 1;
    wp::float32 var_391;
    const wp::float32 var_392 = 1.0;
    const wp::int32 var_393 = 1;
    wp::float32 var_394;
    wp::float32 var_395;
    wp::float32 var_396;
    wp::float32 var_397;
    const wp::int32 var_398 = 4;
    wp::int32 var_399;
    const wp::int32 var_400 = 2;
    wp::float32 var_401;
    const wp::float32 var_402 = 1.0;
    const wp::int32 var_403 = 2;
    wp::float32 var_404;
    wp::float32 var_405;
    wp::float32 var_406;
    wp::float32 var_407;
    const wp::int32 var_408 = 1;
    wp::int32 var_409;
    const wp::int32 var_410 = 0;
    wp::float32 var_411;
    const wp::float32 var_412 = 1.0;
    const wp::int32 var_413 = 0;
    wp::float32 var_414;
    wp::float32 var_415;
    wp::float32 var_416;
    const wp::int32 var_417 = 2;
    wp::int32 var_418;
    const wp::float32 var_419 = 1.0;
    const wp::float32 var_420 = -1.0;
    wp::float32 var_421;
    wp::float32 var_422;
    const wp::int32 var_423 = 4;
    wp::int32 var_424;
    const wp::int32 var_425 = 2;
    wp::float32 var_426;
    const wp::float32 var_427 = 1.0;
    const wp::int32 var_428 = 2;
    wp::float32 var_429;
    wp::float32 var_430;
    wp::float32 var_431;
    wp::float32 var_432;
    const wp::int32 var_433 = 1;
    wp::int32 var_434;
    const wp::int32 var_435 = 0;
    wp::float32 var_436;
    const wp::float32 var_437 = 1.0;
    const wp::int32 var_438 = 0;
    wp::float32 var_439;
    wp::float32 var_440;
    wp::float32 var_441;
    const wp::int32 var_442 = 2;
    wp::int32 var_443;
    const wp::int32 var_444 = 1;
    wp::float32 var_445;
    const wp::float32 var_446 = 1.0;
    const wp::int32 var_447 = 1;
    wp::float32 var_448;
    wp::float32 var_449;
    wp::float32 var_450;
    wp::float32 var_451;
    const wp::int32 var_452 = 4;
    wp::int32 var_453;
    const wp::float32 var_454 = 1.0;
    const wp::float32 var_455 = -1.0;
    wp::float32 var_456;
    wp::float32 var_457;
    const wp::int32 var_458 = 3;
    bool var_459;
    const wp::int32 var_460 = 1;
    wp::int32 var_461;
    const wp::int32 var_462 = 0;
    wp::float32 var_463;
    const wp::float32 var_464 = 1.0;
    const wp::int32 var_465 = 0;
    wp::float32 var_466;
    wp::float32 var_467;
    wp::float32 var_468;
    const wp::int32 var_469 = 2;
    wp::int32 var_470;
    const wp::int32 var_471 = 1;
    wp::float32 var_472;
    const wp::float32 var_473 = 1.0;
    const wp::int32 var_474 = 1;
    wp::float32 var_475;
    wp::float32 var_476;
    wp::float32 var_477;
    wp::float32 var_478;
    const wp::int32 var_479 = 4;
    wp::int32 var_480;
    const wp::int32 var_481 = 2;
    wp::float32 var_482;
    const wp::float32 var_483 = 1.0;
    const wp::int32 var_484 = 2;
    wp::float32 var_485;
    wp::float32 var_486;
    wp::float32 var_487;
    wp::float32 var_488;
    const wp::int32 var_489 = 1;
    wp::int32 var_490;
    const wp::float32 var_491 = 1.0;
    const wp::float32 var_492 = -1.0;
    wp::float32 var_493;
    const wp::int32 var_494 = 2;
    wp::int32 var_495;
    const wp::int32 var_496 = 1;
    wp::float32 var_497;
    const wp::float32 var_498 = 1.0;
    const wp::int32 var_499 = 1;
    wp::float32 var_500;
    wp::float32 var_501;
    wp::float32 var_502;
    wp::float32 var_503;
    const wp::int32 var_504 = 4;
    wp::int32 var_505;
    const wp::int32 var_506 = 2;
    wp::float32 var_507;
    const wp::float32 var_508 = 1.0;
    const wp::int32 var_509 = 2;
    wp::float32 var_510;
    wp::float32 var_511;
    wp::float32 var_512;
    wp::float32 var_513;
    const wp::int32 var_514 = 1;
    wp::int32 var_515;
    const wp::int32 var_516 = 0;
    wp::float32 var_517;
    const wp::float32 var_518 = 1.0;
    const wp::int32 var_519 = 0;
    wp::float32 var_520;
    wp::float32 var_521;
    wp::float32 var_522;
    const wp::int32 var_523 = 2;
    wp::int32 var_524;
    const wp::float32 var_525 = 1.0;
    const wp::float32 var_526 = -1.0;
    wp::float32 var_527;
    wp::float32 var_528;
    const wp::int32 var_529 = 4;
    wp::int32 var_530;
    const wp::int32 var_531 = 2;
    wp::float32 var_532;
    const wp::float32 var_533 = 1.0;
    const wp::int32 var_534 = 2;
    wp::float32 var_535;
    wp::float32 var_536;
    wp::float32 var_537;
    wp::float32 var_538;
    const wp::int32 var_539 = 1;
    wp::int32 var_540;
    const wp::int32 var_541 = 0;
    wp::float32 var_542;
    const wp::float32 var_543 = 1.0;
    const wp::int32 var_544 = 0;
    wp::float32 var_545;
    wp::float32 var_546;
    wp::float32 var_547;
    const wp::int32 var_548 = 2;
    wp::int32 var_549;
    const wp::int32 var_550 = 1;
    wp::float32 var_551;
    const wp::float32 var_552 = 1.0;
    const wp::int32 var_553 = 1;
    wp::float32 var_554;
    wp::float32 var_555;
    wp::float32 var_556;
    wp::float32 var_557;
    const wp::int32 var_558 = 4;
    wp::int32 var_559;
    const wp::float32 var_560 = 1.0;
    const wp::float32 var_561 = -1.0;
    wp::float32 var_562;
    wp::float32 var_563;
    const wp::int32 var_564 = 4;
    bool var_565;
    const wp::int32 var_566 = 1;
    wp::int32 var_567;
    const wp::int32 var_568 = 0;
    wp::float32 var_569;
    const wp::float32 var_570 = 1.0;
    const wp::int32 var_571 = 0;
    wp::float32 var_572;
    wp::float32 var_573;
    wp::float32 var_574;
    const wp::int32 var_575 = 2;
    wp::int32 var_576;
    const wp::int32 var_577 = 1;
    wp::float32 var_578;
    const wp::float32 var_579 = 1.0;
    const wp::int32 var_580 = 1;
    wp::float32 var_581;
    wp::float32 var_582;
    wp::float32 var_583;
    wp::float32 var_584;
    const wp::int32 var_585 = 4;
    wp::int32 var_586;
    const wp::int32 var_587 = 2;
    wp::float32 var_588;
    const wp::float32 var_589 = 1.0;
    const wp::int32 var_590 = 2;
    wp::float32 var_591;
    wp::float32 var_592;
    wp::float32 var_593;
    wp::float32 var_594;
    const wp::int32 var_595 = 1;
    wp::int32 var_596;
    const wp::float32 var_597 = 1.0;
    const wp::float32 var_598 = -1.0;
    wp::float32 var_599;
    const wp::int32 var_600 = 2;
    wp::int32 var_601;
    const wp::int32 var_602 = 1;
    wp::float32 var_603;
    const wp::float32 var_604 = 1.0;
    const wp::int32 var_605 = 1;
    wp::float32 var_606;
    wp::float32 var_607;
    wp::float32 var_608;
    wp::float32 var_609;
    const wp::int32 var_610 = 4;
    wp::int32 var_611;
    const wp::int32 var_612 = 2;
    wp::float32 var_613;
    const wp::float32 var_614 = 1.0;
    const wp::int32 var_615 = 2;
    wp::float32 var_616;
    wp::float32 var_617;
    wp::float32 var_618;
    wp::float32 var_619;
    const wp::int32 var_620 = 1;
    wp::int32 var_621;
    const wp::int32 var_622 = 0;
    wp::float32 var_623;
    const wp::float32 var_624 = 1.0;
    const wp::int32 var_625 = 0;
    wp::float32 var_626;
    wp::float32 var_627;
    wp::float32 var_628;
    const wp::int32 var_629 = 2;
    wp::int32 var_630;
    const wp::float32 var_631 = 1.0;
    const wp::float32 var_632 = -1.0;
    wp::float32 var_633;
    wp::float32 var_634;
    const wp::int32 var_635 = 4;
    wp::int32 var_636;
    const wp::int32 var_637 = 2;
    wp::float32 var_638;
    const wp::float32 var_639 = 1.0;
    const wp::int32 var_640 = 2;
    wp::float32 var_641;
    wp::float32 var_642;
    wp::float32 var_643;
    wp::float32 var_644;
    const wp::int32 var_645 = 1;
    wp::int32 var_646;
    const wp::int32 var_647 = 0;
    wp::float32 var_648;
    const wp::float32 var_649 = 1.0;
    const wp::int32 var_650 = 0;
    wp::float32 var_651;
    wp::float32 var_652;
    wp::float32 var_653;
    const wp::int32 var_654 = 2;
    wp::int32 var_655;
    const wp::int32 var_656 = 1;
    wp::float32 var_657;
    const wp::float32 var_658 = 1.0;
    const wp::int32 var_659 = 1;
    wp::float32 var_660;
    wp::float32 var_661;
    wp::float32 var_662;
    wp::float32 var_663;
    const wp::int32 var_664 = 4;
    wp::int32 var_665;
    const wp::float32 var_666 = 1.0;
    const wp::float32 var_667 = -1.0;
    wp::float32 var_668;
    wp::float32 var_669;
    const wp::int32 var_670 = 5;
    bool var_671;
    const wp::int32 var_672 = 1;
    wp::int32 var_673;
    const wp::int32 var_674 = 0;
    wp::float32 var_675;
    const wp::float32 var_676 = 1.0;
    const wp::int32 var_677 = 0;
    wp::float32 var_678;
    wp::float32 var_679;
    wp::float32 var_680;
    const wp::int32 var_681 = 2;
    wp::int32 var_682;
    const wp::int32 var_683 = 1;
    wp::float32 var_684;
    const wp::float32 var_685 = 1.0;
    const wp::int32 var_686 = 1;
    wp::float32 var_687;
    wp::float32 var_688;
    wp::float32 var_689;
    wp::float32 var_690;
    const wp::int32 var_691 = 4;
    wp::int32 var_692;
    const wp::int32 var_693 = 2;
    wp::float32 var_694;
    const wp::float32 var_695 = 1.0;
    const wp::int32 var_696 = 2;
    wp::float32 var_697;
    wp::float32 var_698;
    wp::float32 var_699;
    wp::float32 var_700;
    const wp::int32 var_701 = 1;
    wp::int32 var_702;
    const wp::float32 var_703 = 1.0;
    const wp::float32 var_704 = -1.0;
    wp::float32 var_705;
    const wp::int32 var_706 = 2;
    wp::int32 var_707;
    const wp::int32 var_708 = 1;
    wp::float32 var_709;
    const wp::float32 var_710 = 1.0;
    const wp::int32 var_711 = 1;
    wp::float32 var_712;
    wp::float32 var_713;
    wp::float32 var_714;
    wp::float32 var_715;
    const wp::int32 var_716 = 4;
    wp::int32 var_717;
    const wp::int32 var_718 = 2;
    wp::float32 var_719;
    const wp::float32 var_720 = 1.0;
    const wp::int32 var_721 = 2;
    wp::float32 var_722;
    wp::float32 var_723;
    wp::float32 var_724;
    wp::float32 var_725;
    const wp::int32 var_726 = 1;
    wp::int32 var_727;
    const wp::int32 var_728 = 0;
    wp::float32 var_729;
    const wp::float32 var_730 = 1.0;
    const wp::int32 var_731 = 0;
    wp::float32 var_732;
    wp::float32 var_733;
    wp::float32 var_734;
    const wp::int32 var_735 = 2;
    wp::int32 var_736;
    const wp::float32 var_737 = 1.0;
    const wp::float32 var_738 = -1.0;
    wp::float32 var_739;
    wp::float32 var_740;
    const wp::int32 var_741 = 4;
    wp::int32 var_742;
    const wp::int32 var_743 = 2;
    wp::float32 var_744;
    const wp::float32 var_745 = 1.0;
    const wp::int32 var_746 = 2;
    wp::float32 var_747;
    wp::float32 var_748;
    wp::float32 var_749;
    wp::float32 var_750;
    const wp::int32 var_751 = 1;
    wp::int32 var_752;
    const wp::int32 var_753 = 0;
    wp::float32 var_754;
    const wp::float32 var_755 = 1.0;
    const wp::int32 var_756 = 0;
    wp::float32 var_757;
    wp::float32 var_758;
    wp::float32 var_759;
    const wp::int32 var_760 = 2;
    wp::int32 var_761;
    const wp::int32 var_762 = 1;
    wp::float32 var_763;
    const wp::float32 var_764 = 1.0;
    const wp::int32 var_765 = 1;
    wp::float32 var_766;
    wp::float32 var_767;
    wp::float32 var_768;
    wp::float32 var_769;
    const wp::int32 var_770 = 4;
    wp::int32 var_771;
    const wp::float32 var_772 = 1.0;
    const wp::float32 var_773 = -1.0;
    wp::float32 var_774;
    wp::float32 var_775;
    const wp::int32 var_776 = 6;
    bool var_777;
    const wp::int32 var_778 = 1;
    wp::int32 var_779;
    const wp::int32 var_780 = 0;
    wp::float32 var_781;
    const wp::float32 var_782 = 1.0;
    const wp::int32 var_783 = 0;
    wp::float32 var_784;
    wp::float32 var_785;
    wp::float32 var_786;
    const wp::int32 var_787 = 2;
    wp::int32 var_788;
    const wp::int32 var_789 = 1;
    wp::float32 var_790;
    const wp::float32 var_791 = 1.0;
    const wp::int32 var_792 = 1;
    wp::float32 var_793;
    wp::float32 var_794;
    wp::float32 var_795;
    wp::float32 var_796;
    const wp::int32 var_797 = 4;
    wp::int32 var_798;
    const wp::int32 var_799 = 2;
    wp::float32 var_800;
    const wp::float32 var_801 = 1.0;
    const wp::int32 var_802 = 2;
    wp::float32 var_803;
    wp::float32 var_804;
    wp::float32 var_805;
    wp::float32 var_806;
    const wp::int32 var_807 = 1;
    wp::int32 var_808;
    const wp::float32 var_809 = 1.0;
    const wp::float32 var_810 = -1.0;
    wp::float32 var_811;
    const wp::int32 var_812 = 2;
    wp::int32 var_813;
    const wp::int32 var_814 = 1;
    wp::float32 var_815;
    const wp::float32 var_816 = 1.0;
    const wp::int32 var_817 = 1;
    wp::float32 var_818;
    wp::float32 var_819;
    wp::float32 var_820;
    wp::float32 var_821;
    const wp::int32 var_822 = 4;
    wp::int32 var_823;
    const wp::int32 var_824 = 2;
    wp::float32 var_825;
    const wp::float32 var_826 = 1.0;
    const wp::int32 var_827 = 2;
    wp::float32 var_828;
    wp::float32 var_829;
    wp::float32 var_830;
    wp::float32 var_831;
    const wp::int32 var_832 = 1;
    wp::int32 var_833;
    const wp::int32 var_834 = 0;
    wp::float32 var_835;
    const wp::float32 var_836 = 1.0;
    const wp::int32 var_837 = 0;
    wp::float32 var_838;
    wp::float32 var_839;
    wp::float32 var_840;
    const wp::int32 var_841 = 2;
    wp::int32 var_842;
    const wp::float32 var_843 = 1.0;
    const wp::float32 var_844 = -1.0;
    wp::float32 var_845;
    wp::float32 var_846;
    const wp::int32 var_847 = 4;
    wp::int32 var_848;
    const wp::int32 var_849 = 2;
    wp::float32 var_850;
    const wp::float32 var_851 = 1.0;
    const wp::int32 var_852 = 2;
    wp::float32 var_853;
    wp::float32 var_854;
    wp::float32 var_855;
    wp::float32 var_856;
    const wp::int32 var_857 = 1;
    wp::int32 var_858;
    const wp::int32 var_859 = 0;
    wp::float32 var_860;
    const wp::float32 var_861 = 1.0;
    const wp::int32 var_862 = 0;
    wp::float32 var_863;
    wp::float32 var_864;
    wp::float32 var_865;
    const wp::int32 var_866 = 2;
    wp::int32 var_867;
    const wp::int32 var_868 = 1;
    wp::float32 var_869;
    const wp::float32 var_870 = 1.0;
    const wp::int32 var_871 = 1;
    wp::float32 var_872;
    wp::float32 var_873;
    wp::float32 var_874;
    wp::float32 var_875;
    const wp::int32 var_876 = 4;
    wp::int32 var_877;
    const wp::float32 var_878 = 1.0;
    const wp::float32 var_879 = -1.0;
    wp::float32 var_880;
    wp::float32 var_881;
    const wp::int32 var_882 = 7;
    bool var_883;
    const wp::int32 var_884 = 1;
    wp::int32 var_885;
    const wp::int32 var_886 = 0;
    wp::float32 var_887;
    const wp::float32 var_888 = 1.0;
    const wp::int32 var_889 = 0;
    wp::float32 var_890;
    wp::float32 var_891;
    wp::float32 var_892;
    const wp::int32 var_893 = 2;
    wp::int32 var_894;
    const wp::int32 var_895 = 1;
    wp::float32 var_896;
    const wp::float32 var_897 = 1.0;
    const wp::int32 var_898 = 1;
    wp::float32 var_899;
    wp::float32 var_900;
    wp::float32 var_901;
    wp::float32 var_902;
    const wp::int32 var_903 = 4;
    wp::int32 var_904;
    const wp::int32 var_905 = 2;
    wp::float32 var_906;
    const wp::float32 var_907 = 1.0;
    const wp::int32 var_908 = 2;
    wp::float32 var_909;
    wp::float32 var_910;
    wp::float32 var_911;
    wp::float32 var_912;
    const wp::int32 var_913 = 1;
    wp::int32 var_914;
    const wp::float32 var_915 = 1.0;
    const wp::float32 var_916 = -1.0;
    wp::float32 var_917;
    const wp::int32 var_918 = 2;
    wp::int32 var_919;
    const wp::int32 var_920 = 1;
    wp::float32 var_921;
    const wp::float32 var_922 = 1.0;
    const wp::int32 var_923 = 1;
    wp::float32 var_924;
    wp::float32 var_925;
    wp::float32 var_926;
    wp::float32 var_927;
    const wp::int32 var_928 = 4;
    wp::int32 var_929;
    const wp::int32 var_930 = 2;
    wp::float32 var_931;
    const wp::float32 var_932 = 1.0;
    const wp::int32 var_933 = 2;
    wp::float32 var_934;
    wp::float32 var_935;
    wp::float32 var_936;
    wp::float32 var_937;
    const wp::int32 var_938 = 1;
    wp::int32 var_939;
    const wp::int32 var_940 = 0;
    wp::float32 var_941;
    const wp::float32 var_942 = 1.0;
    const wp::int32 var_943 = 0;
    wp::float32 var_944;
    wp::float32 var_945;
    wp::float32 var_946;
    const wp::int32 var_947 = 2;
    wp::int32 var_948;
    const wp::float32 var_949 = 1.0;
    const wp::float32 var_950 = -1.0;
    wp::float32 var_951;
    wp::float32 var_952;
    const wp::int32 var_953 = 4;
    wp::int32 var_954;
    const wp::int32 var_955 = 2;
    wp::float32 var_956;
    const wp::float32 var_957 = 1.0;
    const wp::int32 var_958 = 2;
    wp::float32 var_959;
    wp::float32 var_960;
    wp::float32 var_961;
    wp::float32 var_962;
    const wp::int32 var_963 = 1;
    wp::int32 var_964;
    const wp::int32 var_965 = 0;
    wp::float32 var_966;
    const wp::float32 var_967 = 1.0;
    const wp::int32 var_968 = 0;
    wp::float32 var_969;
    wp::float32 var_970;
    wp::float32 var_971;
    const wp::int32 var_972 = 2;
    wp::int32 var_973;
    const wp::int32 var_974 = 1;
    wp::float32 var_975;
    const wp::float32 var_976 = 1.0;
    const wp::int32 var_977 = 1;
    wp::float32 var_978;
    wp::float32 var_979;
    wp::float32 var_980;
    wp::float32 var_981;
    const wp::int32 var_982 = 4;
    wp::int32 var_983;
    const wp::float32 var_984 = 1.0;
    const wp::float32 var_985 = -1.0;
    wp::float32 var_986;
    wp::float32 var_987;
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> var_988;
    const wp::int32 var_989 = 0;
    wp::float32 var_990;
    const wp::float32 var_991 = 0.5;
    bool var_992;
    const wp::int32 var_993 = 0;
    const wp::int32 var_994 = 1;
    wp::int32 var_995;
    const wp::int32 var_996 = 1;
    wp::float32 var_997;
    const wp::float32 var_998 = 0.5;
    bool var_999;
    const wp::int32 var_1000 = 0;
    const wp::int32 var_1001 = 1;
    wp::int32 var_1002;
    const wp::int32 var_1003 = 2;
    wp::float32 var_1004;
    const wp::float32 var_1005 = 0.5;
    bool var_1006;
    const wp::int32 var_1007 = 0;
    const wp::int32 var_1008 = 1;
    wp::int32 var_1009;
    wp::vec_t<8, wp::int32>* var_1010;
    const wp::int32 var_1011 = 4;
    wp::int32 var_1012;
    const wp::int32 var_1013 = 2;
    wp::int32 var_1014;
    wp::int32 var_1015;
    wp::int32 var_1016;
    wp::int32 var_1017;
    wp::vec_t<8, wp::int32> var_1018;
    const wp::int32 var_1019 = -1;
    bool var_1020;
    wp::int32 var_1021;
    const wp::int32 var_1022 = -1;
    wp::int32 var_1023;
    const wp::str var_1024 = "ERROR: Node not found\n";
    const wp::int32 var_1025 = -1;
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> var_1026;
    //---------
    // forward
    // def find_oct(                                                                          <L 319>
    // stack = root                                                                           <L 322>
    var_0 = wp::copy(var_root);
    // niter = int(100)                                                                       <L 323>
    var_2 = wp::int(var_1);
    // rx = vec8(0.0)                                                                         <L 324>
    var_4 = wp::vec_t<8, wp::float32>(var_3);
    // ry = vec8(0.0)                                                                         <L 325>
    var_6 = wp::vec_t<8, wp::float32>(var_5);
    // rz = vec8(0.0)                                                                         <L 326>
    var_8 = wp::vec_t<8, wp::float32>(var_7);
    // eps = 1e-6                                                                             <L 327>
    // while niter > 0:                                                                       <L 329>
    start_while_0:;
    var_11 = (var_2 > var_10);
    if ((var_11) == false) goto end_while_0;
        // niter -= 1                                                                         <L 330>
        var_13 = wp::sub(var_2, var_12);
        // node = stack                                                                       <L 331>
        var_14 = wp::copy(var_0);
        // if node == -1:                                                                     <L 333>
        var_16 = (var_14 == var_15);
        if (var_16) {
            // wp.printf("ERROR: Invalid node number\n")                                      <L 334>
            printf(var_17);
            // return -1, (rx, ry, rz)                                                        <L 335>
            var_19 = wp::tuple(var_4, var_6, var_8);
            ret_0 = var_18;
            ret_1 = var_19;
            return;
        }
        // vmin = oct_aabb[node, 0] - oct_aabb[node, 1]                                       <L 337>
        var_21 = wp::address(var_oct_aabb, var_14, var_20);
        var_23 = wp::address(var_oct_aabb, var_14, var_22);
        var_25 = wp::load(var_21);
        var_26 = wp::load(var_23);
        var_24 = wp::sub(var_25, var_26);
        // vmax = oct_aabb[node, 0] + oct_aabb[node, 1]                                       <L 338>
        var_28 = wp::address(var_oct_aabb, var_14, var_27);
        var_30 = wp::address(var_oct_aabb, var_14, var_29);
        var_32 = wp::load(var_28);
        var_33 = wp::load(var_30);
        var_31 = wp::add(var_32, var_33);
        // if (                                                                               <L 340>
        // p[0] + eps < vmin[0]                                                               <L 341>
        var_36 = wp::extract(var_p, var_35);
        var_37 = wp::add(var_36, var_9);
        var_39 = wp::extract(var_24, var_38);
        var_40 = (var_37 < var_39);
        var_34 = var_40;
        if (!var_34) {
            // or p[0] - eps > vmax[0]                                                        <L 342>
            var_42 = wp::extract(var_p, var_41);
            var_43 = wp::sub(var_42, var_9);
            var_45 = wp::extract(var_31, var_44);
            var_46 = (var_43 > var_45);
            var_34 = var_34 || var_46;
        }
        if (!var_34) {
            // or p[1] + eps < vmin[1]                                                        <L 343>
            var_48 = wp::extract(var_p, var_47);
            var_49 = wp::add(var_48, var_9);
            var_51 = wp::extract(var_24, var_50);
            var_52 = (var_49 < var_51);
            var_34 = var_34 || var_52;
        }
        if (!var_34) {
            // or p[1] - eps > vmax[1]                                                        <L 344>
            var_54 = wp::extract(var_p, var_53);
            var_55 = wp::sub(var_54, var_9);
            var_57 = wp::extract(var_31, var_56);
            var_58 = (var_55 > var_57);
            var_34 = var_34 || var_58;
        }
        if (!var_34) {
            // or p[2] + eps < vmin[2]                                                        <L 345>
            var_60 = wp::extract(var_p, var_59);
            var_61 = wp::add(var_60, var_9);
            var_63 = wp::extract(var_24, var_62);
            var_64 = (var_61 < var_63);
            var_34 = var_34 || var_64;
        }
        if (!var_34) {
            // or p[2] - eps > vmax[2]                                                        <L 346>
            var_66 = wp::extract(var_p, var_65);
            var_67 = wp::sub(var_66, var_9);
            var_69 = wp::extract(var_31, var_68);
            var_70 = (var_67 > var_69);
            var_34 = var_34 || var_70;
        }
        if (var_34) {
            // continue                                                                       <L 348>
            wp::assign(var_2, var_13);
            goto start_while_0;
        }
        var_71 = wp::where(var_34, var_2, var_13);
        // coord = wp.cw_div(p - vmin, vmax - vmin)                                           <L 350>
        var_72 = wp::sub(var_p, var_24);
        var_73 = wp::sub(var_31, var_24);
        var_74 = wp::cw_div(var_72, var_73);
        // child0 = oct_child[node][0]                                                        <L 354>
        var_75 = wp::address(var_oct_child, var_14);
        var_78 = wp::load(var_75);
        var_77 = wp::extract(var_78, var_76);
        // if (                                                                               <L 356>
        // int(child0 == -1)                                                                  <L 357>
        var_80 = (var_77 == var_79);
        var_81 = wp::int(var_80);
        // & int(oct_child[node][1] == -1)                                                    <L 358>
        var_82 = wp::address(var_oct_child, var_14);
        var_85 = wp::load(var_82);
        var_84 = wp::extract(var_85, var_83);
        var_87 = (var_84 == var_86);
        var_88 = wp::int(var_87);
        var_89 = wp::bit_and(var_81, var_88);
        // & int(oct_child[node][2] == -1)                                                    <L 359>
        var_90 = wp::address(var_oct_child, var_14);
        var_93 = wp::load(var_90);
        var_92 = wp::extract(var_93, var_91);
        var_95 = (var_92 == var_94);
        var_96 = wp::int(var_95);
        var_97 = wp::bit_and(var_89, var_96);
        // & int(oct_child[node][3] == -1)                                                    <L 360>
        var_98 = wp::address(var_oct_child, var_14);
        var_101 = wp::load(var_98);
        var_100 = wp::extract(var_101, var_99);
        var_103 = (var_100 == var_102);
        var_104 = wp::int(var_103);
        var_105 = wp::bit_and(var_97, var_104);
        // & int(oct_child[node][4] == -1)                                                    <L 361>
        var_106 = wp::address(var_oct_child, var_14);
        var_109 = wp::load(var_106);
        var_108 = wp::extract(var_109, var_107);
        var_111 = (var_108 == var_110);
        var_112 = wp::int(var_111);
        var_113 = wp::bit_and(var_105, var_112);
        // & int(oct_child[node][5] == -1)                                                    <L 362>
        var_114 = wp::address(var_oct_child, var_14);
        var_117 = wp::load(var_114);
        var_116 = wp::extract(var_117, var_115);
        var_119 = (var_116 == var_118);
        var_120 = wp::int(var_119);
        var_121 = wp::bit_and(var_113, var_120);
        // & int(oct_child[node][6] == -1)                                                    <L 363>
        var_122 = wp::address(var_oct_child, var_14);
        var_125 = wp::load(var_122);
        var_124 = wp::extract(var_125, var_123);
        var_127 = (var_124 == var_126);
        var_128 = wp::int(var_127);
        var_129 = wp::bit_and(var_121, var_128);
        // & int(oct_child[node][7] == -1)                                                    <L 364>
        var_130 = wp::address(var_oct_child, var_14);
        var_133 = wp::load(var_130);
        var_132 = wp::extract(var_133, var_131);
        var_135 = (var_132 == var_134);
        var_136 = wp::int(var_135);
        var_137 = wp::bit_and(var_129, var_136);
        // ) != 0:                                                                            <L 365>
        var_139 = (var_137 != var_138);
        if (var_139) {
            // for j in range(8):                                                             <L 366>
            // if not grad:                                                                   <L 367>
            var_141 = wp::unot(var_grad);
            if (var_141) {
                // rx[j] = (                                                                  <L 368>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 369>
                var_143 = wp::bit_and(var_140, var_142);
                if (var_143) {
                    var_145 = wp::extract(var_74, var_144);
                }
                if (!var_143) {
                    var_148 = wp::extract(var_74, var_147);
                    var_149 = wp::sub(var_146, var_148);
                }
                var_150 = wp::where(var_143, var_145, var_149);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 370>
                var_152 = wp::bit_and(var_140, var_151);
                if (var_152) {
                    var_154 = wp::extract(var_74, var_153);
                }
                if (!var_152) {
                    var_157 = wp::extract(var_74, var_156);
                    var_158 = wp::sub(var_155, var_157);
                }
                var_159 = wp::where(var_152, var_154, var_158);
                var_160 = wp::mul(var_150, var_159);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 371>
                var_162 = wp::bit_and(var_140, var_161);
                if (var_162) {
                    var_164 = wp::extract(var_74, var_163);
                }
                if (!var_162) {
                    var_167 = wp::extract(var_74, var_166);
                    var_168 = wp::sub(var_165, var_167);
                }
                var_169 = wp::where(var_162, var_164, var_168);
                var_170 = wp::mul(var_160, var_169);
                // rx[j] = (                                                                  <L 368>
                wp::assign_inplace(var_4, var_140, var_170);
            }
            if (!var_141) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 374>
                var_172 = wp::bit_and(var_140, var_171);
                if (var_172) {
                }
                if (!var_172) {
                }
                var_175 = wp::where(var_172, var_173, var_174);
                var_177 = wp::bit_and(var_140, var_176);
                if (var_177) {
                    var_179 = wp::extract(var_74, var_178);
                }
                if (!var_177) {
                    var_182 = wp::extract(var_74, var_181);
                    var_183 = wp::sub(var_180, var_182);
                }
                var_184 = wp::where(var_177, var_179, var_183);
                var_185 = wp::mul(var_175, var_184);
                var_187 = wp::bit_and(var_140, var_186);
                if (var_187) {
                    var_189 = wp::extract(var_74, var_188);
                }
                if (!var_187) {
                    var_192 = wp::extract(var_74, var_191);
                    var_193 = wp::sub(var_190, var_192);
                }
                var_194 = wp::where(var_187, var_189, var_193);
                var_195 = wp::mul(var_185, var_194);
                wp::assign_inplace(var_4, var_140, var_195);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 375>
                var_197 = wp::bit_and(var_140, var_196);
                if (var_197) {
                    var_199 = wp::extract(var_74, var_198);
                }
                if (!var_197) {
                    var_202 = wp::extract(var_74, var_201);
                    var_203 = wp::sub(var_200, var_202);
                }
                var_204 = wp::where(var_197, var_199, var_203);
                var_206 = wp::bit_and(var_140, var_205);
                if (var_206) {
                }
                if (!var_206) {
                }
                var_209 = wp::where(var_206, var_207, var_208);
                var_210 = wp::mul(var_204, var_209);
                var_212 = wp::bit_and(var_140, var_211);
                if (var_212) {
                    var_214 = wp::extract(var_74, var_213);
                }
                if (!var_212) {
                    var_217 = wp::extract(var_74, var_216);
                    var_218 = wp::sub(var_215, var_217);
                }
                var_219 = wp::where(var_212, var_214, var_218);
                var_220 = wp::mul(var_210, var_219);
                wp::assign_inplace(var_6, var_140, var_220);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 376>
                var_222 = wp::bit_and(var_140, var_221);
                if (var_222) {
                    var_224 = wp::extract(var_74, var_223);
                }
                if (!var_222) {
                    var_227 = wp::extract(var_74, var_226);
                    var_228 = wp::sub(var_225, var_227);
                }
                var_229 = wp::where(var_222, var_224, var_228);
                var_231 = wp::bit_and(var_140, var_230);
                if (var_231) {
                    var_233 = wp::extract(var_74, var_232);
                }
                if (!var_231) {
                    var_236 = wp::extract(var_74, var_235);
                    var_237 = wp::sub(var_234, var_236);
                }
                var_238 = wp::where(var_231, var_233, var_237);
                var_239 = wp::mul(var_229, var_238);
                var_241 = wp::bit_and(var_140, var_240);
                if (var_241) {
                }
                if (!var_241) {
                }
                var_244 = wp::where(var_241, var_242, var_243);
                var_245 = wp::mul(var_239, var_244);
                wp::assign_inplace(var_8, var_140, var_245);
            }
            // if not grad:                                                                   <L 367>
            var_247 = wp::unot(var_grad);
            if (var_247) {
                // rx[j] = (                                                                  <L 368>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 369>
                var_249 = wp::bit_and(var_246, var_248);
                if (var_249) {
                    var_251 = wp::extract(var_74, var_250);
                }
                if (!var_249) {
                    var_254 = wp::extract(var_74, var_253);
                    var_255 = wp::sub(var_252, var_254);
                }
                var_256 = wp::where(var_249, var_251, var_255);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 370>
                var_258 = wp::bit_and(var_246, var_257);
                if (var_258) {
                    var_260 = wp::extract(var_74, var_259);
                }
                if (!var_258) {
                    var_263 = wp::extract(var_74, var_262);
                    var_264 = wp::sub(var_261, var_263);
                }
                var_265 = wp::where(var_258, var_260, var_264);
                var_266 = wp::mul(var_256, var_265);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 371>
                var_268 = wp::bit_and(var_246, var_267);
                if (var_268) {
                    var_270 = wp::extract(var_74, var_269);
                }
                if (!var_268) {
                    var_273 = wp::extract(var_74, var_272);
                    var_274 = wp::sub(var_271, var_273);
                }
                var_275 = wp::where(var_268, var_270, var_274);
                var_276 = wp::mul(var_266, var_275);
                // rx[j] = (                                                                  <L 368>
                wp::assign_inplace(var_4, var_246, var_276);
            }
            if (!var_247) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 374>
                var_278 = wp::bit_and(var_246, var_277);
                if (var_278) {
                }
                if (!var_278) {
                }
                var_281 = wp::where(var_278, var_279, var_280);
                var_283 = wp::bit_and(var_246, var_282);
                if (var_283) {
                    var_285 = wp::extract(var_74, var_284);
                }
                if (!var_283) {
                    var_288 = wp::extract(var_74, var_287);
                    var_289 = wp::sub(var_286, var_288);
                }
                var_290 = wp::where(var_283, var_285, var_289);
                var_291 = wp::mul(var_281, var_290);
                var_293 = wp::bit_and(var_246, var_292);
                if (var_293) {
                    var_295 = wp::extract(var_74, var_294);
                }
                if (!var_293) {
                    var_298 = wp::extract(var_74, var_297);
                    var_299 = wp::sub(var_296, var_298);
                }
                var_300 = wp::where(var_293, var_295, var_299);
                var_301 = wp::mul(var_291, var_300);
                wp::assign_inplace(var_4, var_246, var_301);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 375>
                var_303 = wp::bit_and(var_246, var_302);
                if (var_303) {
                    var_305 = wp::extract(var_74, var_304);
                }
                if (!var_303) {
                    var_308 = wp::extract(var_74, var_307);
                    var_309 = wp::sub(var_306, var_308);
                }
                var_310 = wp::where(var_303, var_305, var_309);
                var_312 = wp::bit_and(var_246, var_311);
                if (var_312) {
                }
                if (!var_312) {
                }
                var_315 = wp::where(var_312, var_313, var_314);
                var_316 = wp::mul(var_310, var_315);
                var_318 = wp::bit_and(var_246, var_317);
                if (var_318) {
                    var_320 = wp::extract(var_74, var_319);
                }
                if (!var_318) {
                    var_323 = wp::extract(var_74, var_322);
                    var_324 = wp::sub(var_321, var_323);
                }
                var_325 = wp::where(var_318, var_320, var_324);
                var_326 = wp::mul(var_316, var_325);
                wp::assign_inplace(var_6, var_246, var_326);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 376>
                var_328 = wp::bit_and(var_246, var_327);
                if (var_328) {
                    var_330 = wp::extract(var_74, var_329);
                }
                if (!var_328) {
                    var_333 = wp::extract(var_74, var_332);
                    var_334 = wp::sub(var_331, var_333);
                }
                var_335 = wp::where(var_328, var_330, var_334);
                var_337 = wp::bit_and(var_246, var_336);
                if (var_337) {
                    var_339 = wp::extract(var_74, var_338);
                }
                if (!var_337) {
                    var_342 = wp::extract(var_74, var_341);
                    var_343 = wp::sub(var_340, var_342);
                }
                var_344 = wp::where(var_337, var_339, var_343);
                var_345 = wp::mul(var_335, var_344);
                var_347 = wp::bit_and(var_246, var_346);
                if (var_347) {
                }
                if (!var_347) {
                }
                var_350 = wp::where(var_347, var_348, var_349);
                var_351 = wp::mul(var_345, var_350);
                wp::assign_inplace(var_8, var_246, var_351);
            }
            // if not grad:                                                                   <L 367>
            var_353 = wp::unot(var_grad);
            if (var_353) {
                // rx[j] = (                                                                  <L 368>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 369>
                var_355 = wp::bit_and(var_352, var_354);
                if (var_355) {
                    var_357 = wp::extract(var_74, var_356);
                }
                if (!var_355) {
                    var_360 = wp::extract(var_74, var_359);
                    var_361 = wp::sub(var_358, var_360);
                }
                var_362 = wp::where(var_355, var_357, var_361);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 370>
                var_364 = wp::bit_and(var_352, var_363);
                if (var_364) {
                    var_366 = wp::extract(var_74, var_365);
                }
                if (!var_364) {
                    var_369 = wp::extract(var_74, var_368);
                    var_370 = wp::sub(var_367, var_369);
                }
                var_371 = wp::where(var_364, var_366, var_370);
                var_372 = wp::mul(var_362, var_371);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 371>
                var_374 = wp::bit_and(var_352, var_373);
                if (var_374) {
                    var_376 = wp::extract(var_74, var_375);
                }
                if (!var_374) {
                    var_379 = wp::extract(var_74, var_378);
                    var_380 = wp::sub(var_377, var_379);
                }
                var_381 = wp::where(var_374, var_376, var_380);
                var_382 = wp::mul(var_372, var_381);
                // rx[j] = (                                                                  <L 368>
                wp::assign_inplace(var_4, var_352, var_382);
            }
            if (!var_353) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 374>
                var_384 = wp::bit_and(var_352, var_383);
                if (var_384) {
                }
                if (!var_384) {
                }
                var_387 = wp::where(var_384, var_385, var_386);
                var_389 = wp::bit_and(var_352, var_388);
                if (var_389) {
                    var_391 = wp::extract(var_74, var_390);
                }
                if (!var_389) {
                    var_394 = wp::extract(var_74, var_393);
                    var_395 = wp::sub(var_392, var_394);
                }
                var_396 = wp::where(var_389, var_391, var_395);
                var_397 = wp::mul(var_387, var_396);
                var_399 = wp::bit_and(var_352, var_398);
                if (var_399) {
                    var_401 = wp::extract(var_74, var_400);
                }
                if (!var_399) {
                    var_404 = wp::extract(var_74, var_403);
                    var_405 = wp::sub(var_402, var_404);
                }
                var_406 = wp::where(var_399, var_401, var_405);
                var_407 = wp::mul(var_397, var_406);
                wp::assign_inplace(var_4, var_352, var_407);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 375>
                var_409 = wp::bit_and(var_352, var_408);
                if (var_409) {
                    var_411 = wp::extract(var_74, var_410);
                }
                if (!var_409) {
                    var_414 = wp::extract(var_74, var_413);
                    var_415 = wp::sub(var_412, var_414);
                }
                var_416 = wp::where(var_409, var_411, var_415);
                var_418 = wp::bit_and(var_352, var_417);
                if (var_418) {
                }
                if (!var_418) {
                }
                var_421 = wp::where(var_418, var_419, var_420);
                var_422 = wp::mul(var_416, var_421);
                var_424 = wp::bit_and(var_352, var_423);
                if (var_424) {
                    var_426 = wp::extract(var_74, var_425);
                }
                if (!var_424) {
                    var_429 = wp::extract(var_74, var_428);
                    var_430 = wp::sub(var_427, var_429);
                }
                var_431 = wp::where(var_424, var_426, var_430);
                var_432 = wp::mul(var_422, var_431);
                wp::assign_inplace(var_6, var_352, var_432);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 376>
                var_434 = wp::bit_and(var_352, var_433);
                if (var_434) {
                    var_436 = wp::extract(var_74, var_435);
                }
                if (!var_434) {
                    var_439 = wp::extract(var_74, var_438);
                    var_440 = wp::sub(var_437, var_439);
                }
                var_441 = wp::where(var_434, var_436, var_440);
                var_443 = wp::bit_and(var_352, var_442);
                if (var_443) {
                    var_445 = wp::extract(var_74, var_444);
                }
                if (!var_443) {
                    var_448 = wp::extract(var_74, var_447);
                    var_449 = wp::sub(var_446, var_448);
                }
                var_450 = wp::where(var_443, var_445, var_449);
                var_451 = wp::mul(var_441, var_450);
                var_453 = wp::bit_and(var_352, var_452);
                if (var_453) {
                }
                if (!var_453) {
                }
                var_456 = wp::where(var_453, var_454, var_455);
                var_457 = wp::mul(var_451, var_456);
                wp::assign_inplace(var_8, var_352, var_457);
            }
            // if not grad:                                                                   <L 367>
            var_459 = wp::unot(var_grad);
            if (var_459) {
                // rx[j] = (                                                                  <L 368>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 369>
                var_461 = wp::bit_and(var_458, var_460);
                if (var_461) {
                    var_463 = wp::extract(var_74, var_462);
                }
                if (!var_461) {
                    var_466 = wp::extract(var_74, var_465);
                    var_467 = wp::sub(var_464, var_466);
                }
                var_468 = wp::where(var_461, var_463, var_467);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 370>
                var_470 = wp::bit_and(var_458, var_469);
                if (var_470) {
                    var_472 = wp::extract(var_74, var_471);
                }
                if (!var_470) {
                    var_475 = wp::extract(var_74, var_474);
                    var_476 = wp::sub(var_473, var_475);
                }
                var_477 = wp::where(var_470, var_472, var_476);
                var_478 = wp::mul(var_468, var_477);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 371>
                var_480 = wp::bit_and(var_458, var_479);
                if (var_480) {
                    var_482 = wp::extract(var_74, var_481);
                }
                if (!var_480) {
                    var_485 = wp::extract(var_74, var_484);
                    var_486 = wp::sub(var_483, var_485);
                }
                var_487 = wp::where(var_480, var_482, var_486);
                var_488 = wp::mul(var_478, var_487);
                // rx[j] = (                                                                  <L 368>
                wp::assign_inplace(var_4, var_458, var_488);
            }
            if (!var_459) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 374>
                var_490 = wp::bit_and(var_458, var_489);
                if (var_490) {
                }
                if (!var_490) {
                }
                var_493 = wp::where(var_490, var_491, var_492);
                var_495 = wp::bit_and(var_458, var_494);
                if (var_495) {
                    var_497 = wp::extract(var_74, var_496);
                }
                if (!var_495) {
                    var_500 = wp::extract(var_74, var_499);
                    var_501 = wp::sub(var_498, var_500);
                }
                var_502 = wp::where(var_495, var_497, var_501);
                var_503 = wp::mul(var_493, var_502);
                var_505 = wp::bit_and(var_458, var_504);
                if (var_505) {
                    var_507 = wp::extract(var_74, var_506);
                }
                if (!var_505) {
                    var_510 = wp::extract(var_74, var_509);
                    var_511 = wp::sub(var_508, var_510);
                }
                var_512 = wp::where(var_505, var_507, var_511);
                var_513 = wp::mul(var_503, var_512);
                wp::assign_inplace(var_4, var_458, var_513);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 375>
                var_515 = wp::bit_and(var_458, var_514);
                if (var_515) {
                    var_517 = wp::extract(var_74, var_516);
                }
                if (!var_515) {
                    var_520 = wp::extract(var_74, var_519);
                    var_521 = wp::sub(var_518, var_520);
                }
                var_522 = wp::where(var_515, var_517, var_521);
                var_524 = wp::bit_and(var_458, var_523);
                if (var_524) {
                }
                if (!var_524) {
                }
                var_527 = wp::where(var_524, var_525, var_526);
                var_528 = wp::mul(var_522, var_527);
                var_530 = wp::bit_and(var_458, var_529);
                if (var_530) {
                    var_532 = wp::extract(var_74, var_531);
                }
                if (!var_530) {
                    var_535 = wp::extract(var_74, var_534);
                    var_536 = wp::sub(var_533, var_535);
                }
                var_537 = wp::where(var_530, var_532, var_536);
                var_538 = wp::mul(var_528, var_537);
                wp::assign_inplace(var_6, var_458, var_538);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 376>
                var_540 = wp::bit_and(var_458, var_539);
                if (var_540) {
                    var_542 = wp::extract(var_74, var_541);
                }
                if (!var_540) {
                    var_545 = wp::extract(var_74, var_544);
                    var_546 = wp::sub(var_543, var_545);
                }
                var_547 = wp::where(var_540, var_542, var_546);
                var_549 = wp::bit_and(var_458, var_548);
                if (var_549) {
                    var_551 = wp::extract(var_74, var_550);
                }
                if (!var_549) {
                    var_554 = wp::extract(var_74, var_553);
                    var_555 = wp::sub(var_552, var_554);
                }
                var_556 = wp::where(var_549, var_551, var_555);
                var_557 = wp::mul(var_547, var_556);
                var_559 = wp::bit_and(var_458, var_558);
                if (var_559) {
                }
                if (!var_559) {
                }
                var_562 = wp::where(var_559, var_560, var_561);
                var_563 = wp::mul(var_557, var_562);
                wp::assign_inplace(var_8, var_458, var_563);
            }
            // if not grad:                                                                   <L 367>
            var_565 = wp::unot(var_grad);
            if (var_565) {
                // rx[j] = (                                                                  <L 368>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 369>
                var_567 = wp::bit_and(var_564, var_566);
                if (var_567) {
                    var_569 = wp::extract(var_74, var_568);
                }
                if (!var_567) {
                    var_572 = wp::extract(var_74, var_571);
                    var_573 = wp::sub(var_570, var_572);
                }
                var_574 = wp::where(var_567, var_569, var_573);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 370>
                var_576 = wp::bit_and(var_564, var_575);
                if (var_576) {
                    var_578 = wp::extract(var_74, var_577);
                }
                if (!var_576) {
                    var_581 = wp::extract(var_74, var_580);
                    var_582 = wp::sub(var_579, var_581);
                }
                var_583 = wp::where(var_576, var_578, var_582);
                var_584 = wp::mul(var_574, var_583);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 371>
                var_586 = wp::bit_and(var_564, var_585);
                if (var_586) {
                    var_588 = wp::extract(var_74, var_587);
                }
                if (!var_586) {
                    var_591 = wp::extract(var_74, var_590);
                    var_592 = wp::sub(var_589, var_591);
                }
                var_593 = wp::where(var_586, var_588, var_592);
                var_594 = wp::mul(var_584, var_593);
                // rx[j] = (                                                                  <L 368>
                wp::assign_inplace(var_4, var_564, var_594);
            }
            if (!var_565) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 374>
                var_596 = wp::bit_and(var_564, var_595);
                if (var_596) {
                }
                if (!var_596) {
                }
                var_599 = wp::where(var_596, var_597, var_598);
                var_601 = wp::bit_and(var_564, var_600);
                if (var_601) {
                    var_603 = wp::extract(var_74, var_602);
                }
                if (!var_601) {
                    var_606 = wp::extract(var_74, var_605);
                    var_607 = wp::sub(var_604, var_606);
                }
                var_608 = wp::where(var_601, var_603, var_607);
                var_609 = wp::mul(var_599, var_608);
                var_611 = wp::bit_and(var_564, var_610);
                if (var_611) {
                    var_613 = wp::extract(var_74, var_612);
                }
                if (!var_611) {
                    var_616 = wp::extract(var_74, var_615);
                    var_617 = wp::sub(var_614, var_616);
                }
                var_618 = wp::where(var_611, var_613, var_617);
                var_619 = wp::mul(var_609, var_618);
                wp::assign_inplace(var_4, var_564, var_619);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 375>
                var_621 = wp::bit_and(var_564, var_620);
                if (var_621) {
                    var_623 = wp::extract(var_74, var_622);
                }
                if (!var_621) {
                    var_626 = wp::extract(var_74, var_625);
                    var_627 = wp::sub(var_624, var_626);
                }
                var_628 = wp::where(var_621, var_623, var_627);
                var_630 = wp::bit_and(var_564, var_629);
                if (var_630) {
                }
                if (!var_630) {
                }
                var_633 = wp::where(var_630, var_631, var_632);
                var_634 = wp::mul(var_628, var_633);
                var_636 = wp::bit_and(var_564, var_635);
                if (var_636) {
                    var_638 = wp::extract(var_74, var_637);
                }
                if (!var_636) {
                    var_641 = wp::extract(var_74, var_640);
                    var_642 = wp::sub(var_639, var_641);
                }
                var_643 = wp::where(var_636, var_638, var_642);
                var_644 = wp::mul(var_634, var_643);
                wp::assign_inplace(var_6, var_564, var_644);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 376>
                var_646 = wp::bit_and(var_564, var_645);
                if (var_646) {
                    var_648 = wp::extract(var_74, var_647);
                }
                if (!var_646) {
                    var_651 = wp::extract(var_74, var_650);
                    var_652 = wp::sub(var_649, var_651);
                }
                var_653 = wp::where(var_646, var_648, var_652);
                var_655 = wp::bit_and(var_564, var_654);
                if (var_655) {
                    var_657 = wp::extract(var_74, var_656);
                }
                if (!var_655) {
                    var_660 = wp::extract(var_74, var_659);
                    var_661 = wp::sub(var_658, var_660);
                }
                var_662 = wp::where(var_655, var_657, var_661);
                var_663 = wp::mul(var_653, var_662);
                var_665 = wp::bit_and(var_564, var_664);
                if (var_665) {
                }
                if (!var_665) {
                }
                var_668 = wp::where(var_665, var_666, var_667);
                var_669 = wp::mul(var_663, var_668);
                wp::assign_inplace(var_8, var_564, var_669);
            }
            // if not grad:                                                                   <L 367>
            var_671 = wp::unot(var_grad);
            if (var_671) {
                // rx[j] = (                                                                  <L 368>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 369>
                var_673 = wp::bit_and(var_670, var_672);
                if (var_673) {
                    var_675 = wp::extract(var_74, var_674);
                }
                if (!var_673) {
                    var_678 = wp::extract(var_74, var_677);
                    var_679 = wp::sub(var_676, var_678);
                }
                var_680 = wp::where(var_673, var_675, var_679);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 370>
                var_682 = wp::bit_and(var_670, var_681);
                if (var_682) {
                    var_684 = wp::extract(var_74, var_683);
                }
                if (!var_682) {
                    var_687 = wp::extract(var_74, var_686);
                    var_688 = wp::sub(var_685, var_687);
                }
                var_689 = wp::where(var_682, var_684, var_688);
                var_690 = wp::mul(var_680, var_689);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 371>
                var_692 = wp::bit_and(var_670, var_691);
                if (var_692) {
                    var_694 = wp::extract(var_74, var_693);
                }
                if (!var_692) {
                    var_697 = wp::extract(var_74, var_696);
                    var_698 = wp::sub(var_695, var_697);
                }
                var_699 = wp::where(var_692, var_694, var_698);
                var_700 = wp::mul(var_690, var_699);
                // rx[j] = (                                                                  <L 368>
                wp::assign_inplace(var_4, var_670, var_700);
            }
            if (!var_671) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 374>
                var_702 = wp::bit_and(var_670, var_701);
                if (var_702) {
                }
                if (!var_702) {
                }
                var_705 = wp::where(var_702, var_703, var_704);
                var_707 = wp::bit_and(var_670, var_706);
                if (var_707) {
                    var_709 = wp::extract(var_74, var_708);
                }
                if (!var_707) {
                    var_712 = wp::extract(var_74, var_711);
                    var_713 = wp::sub(var_710, var_712);
                }
                var_714 = wp::where(var_707, var_709, var_713);
                var_715 = wp::mul(var_705, var_714);
                var_717 = wp::bit_and(var_670, var_716);
                if (var_717) {
                    var_719 = wp::extract(var_74, var_718);
                }
                if (!var_717) {
                    var_722 = wp::extract(var_74, var_721);
                    var_723 = wp::sub(var_720, var_722);
                }
                var_724 = wp::where(var_717, var_719, var_723);
                var_725 = wp::mul(var_715, var_724);
                wp::assign_inplace(var_4, var_670, var_725);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 375>
                var_727 = wp::bit_and(var_670, var_726);
                if (var_727) {
                    var_729 = wp::extract(var_74, var_728);
                }
                if (!var_727) {
                    var_732 = wp::extract(var_74, var_731);
                    var_733 = wp::sub(var_730, var_732);
                }
                var_734 = wp::where(var_727, var_729, var_733);
                var_736 = wp::bit_and(var_670, var_735);
                if (var_736) {
                }
                if (!var_736) {
                }
                var_739 = wp::where(var_736, var_737, var_738);
                var_740 = wp::mul(var_734, var_739);
                var_742 = wp::bit_and(var_670, var_741);
                if (var_742) {
                    var_744 = wp::extract(var_74, var_743);
                }
                if (!var_742) {
                    var_747 = wp::extract(var_74, var_746);
                    var_748 = wp::sub(var_745, var_747);
                }
                var_749 = wp::where(var_742, var_744, var_748);
                var_750 = wp::mul(var_740, var_749);
                wp::assign_inplace(var_6, var_670, var_750);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 376>
                var_752 = wp::bit_and(var_670, var_751);
                if (var_752) {
                    var_754 = wp::extract(var_74, var_753);
                }
                if (!var_752) {
                    var_757 = wp::extract(var_74, var_756);
                    var_758 = wp::sub(var_755, var_757);
                }
                var_759 = wp::where(var_752, var_754, var_758);
                var_761 = wp::bit_and(var_670, var_760);
                if (var_761) {
                    var_763 = wp::extract(var_74, var_762);
                }
                if (!var_761) {
                    var_766 = wp::extract(var_74, var_765);
                    var_767 = wp::sub(var_764, var_766);
                }
                var_768 = wp::where(var_761, var_763, var_767);
                var_769 = wp::mul(var_759, var_768);
                var_771 = wp::bit_and(var_670, var_770);
                if (var_771) {
                }
                if (!var_771) {
                }
                var_774 = wp::where(var_771, var_772, var_773);
                var_775 = wp::mul(var_769, var_774);
                wp::assign_inplace(var_8, var_670, var_775);
            }
            // if not grad:                                                                   <L 367>
            var_777 = wp::unot(var_grad);
            if (var_777) {
                // rx[j] = (                                                                  <L 368>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 369>
                var_779 = wp::bit_and(var_776, var_778);
                if (var_779) {
                    var_781 = wp::extract(var_74, var_780);
                }
                if (!var_779) {
                    var_784 = wp::extract(var_74, var_783);
                    var_785 = wp::sub(var_782, var_784);
                }
                var_786 = wp::where(var_779, var_781, var_785);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 370>
                var_788 = wp::bit_and(var_776, var_787);
                if (var_788) {
                    var_790 = wp::extract(var_74, var_789);
                }
                if (!var_788) {
                    var_793 = wp::extract(var_74, var_792);
                    var_794 = wp::sub(var_791, var_793);
                }
                var_795 = wp::where(var_788, var_790, var_794);
                var_796 = wp::mul(var_786, var_795);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 371>
                var_798 = wp::bit_and(var_776, var_797);
                if (var_798) {
                    var_800 = wp::extract(var_74, var_799);
                }
                if (!var_798) {
                    var_803 = wp::extract(var_74, var_802);
                    var_804 = wp::sub(var_801, var_803);
                }
                var_805 = wp::where(var_798, var_800, var_804);
                var_806 = wp::mul(var_796, var_805);
                // rx[j] = (                                                                  <L 368>
                wp::assign_inplace(var_4, var_776, var_806);
            }
            if (!var_777) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 374>
                var_808 = wp::bit_and(var_776, var_807);
                if (var_808) {
                }
                if (!var_808) {
                }
                var_811 = wp::where(var_808, var_809, var_810);
                var_813 = wp::bit_and(var_776, var_812);
                if (var_813) {
                    var_815 = wp::extract(var_74, var_814);
                }
                if (!var_813) {
                    var_818 = wp::extract(var_74, var_817);
                    var_819 = wp::sub(var_816, var_818);
                }
                var_820 = wp::where(var_813, var_815, var_819);
                var_821 = wp::mul(var_811, var_820);
                var_823 = wp::bit_and(var_776, var_822);
                if (var_823) {
                    var_825 = wp::extract(var_74, var_824);
                }
                if (!var_823) {
                    var_828 = wp::extract(var_74, var_827);
                    var_829 = wp::sub(var_826, var_828);
                }
                var_830 = wp::where(var_823, var_825, var_829);
                var_831 = wp::mul(var_821, var_830);
                wp::assign_inplace(var_4, var_776, var_831);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 375>
                var_833 = wp::bit_and(var_776, var_832);
                if (var_833) {
                    var_835 = wp::extract(var_74, var_834);
                }
                if (!var_833) {
                    var_838 = wp::extract(var_74, var_837);
                    var_839 = wp::sub(var_836, var_838);
                }
                var_840 = wp::where(var_833, var_835, var_839);
                var_842 = wp::bit_and(var_776, var_841);
                if (var_842) {
                }
                if (!var_842) {
                }
                var_845 = wp::where(var_842, var_843, var_844);
                var_846 = wp::mul(var_840, var_845);
                var_848 = wp::bit_and(var_776, var_847);
                if (var_848) {
                    var_850 = wp::extract(var_74, var_849);
                }
                if (!var_848) {
                    var_853 = wp::extract(var_74, var_852);
                    var_854 = wp::sub(var_851, var_853);
                }
                var_855 = wp::where(var_848, var_850, var_854);
                var_856 = wp::mul(var_846, var_855);
                wp::assign_inplace(var_6, var_776, var_856);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 376>
                var_858 = wp::bit_and(var_776, var_857);
                if (var_858) {
                    var_860 = wp::extract(var_74, var_859);
                }
                if (!var_858) {
                    var_863 = wp::extract(var_74, var_862);
                    var_864 = wp::sub(var_861, var_863);
                }
                var_865 = wp::where(var_858, var_860, var_864);
                var_867 = wp::bit_and(var_776, var_866);
                if (var_867) {
                    var_869 = wp::extract(var_74, var_868);
                }
                if (!var_867) {
                    var_872 = wp::extract(var_74, var_871);
                    var_873 = wp::sub(var_870, var_872);
                }
                var_874 = wp::where(var_867, var_869, var_873);
                var_875 = wp::mul(var_865, var_874);
                var_877 = wp::bit_and(var_776, var_876);
                if (var_877) {
                }
                if (!var_877) {
                }
                var_880 = wp::where(var_877, var_878, var_879);
                var_881 = wp::mul(var_875, var_880);
                wp::assign_inplace(var_8, var_776, var_881);
            }
            // if not grad:                                                                   <L 367>
            var_883 = wp::unot(var_grad);
            if (var_883) {
                // rx[j] = (                                                                  <L 368>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 369>
                var_885 = wp::bit_and(var_882, var_884);
                if (var_885) {
                    var_887 = wp::extract(var_74, var_886);
                }
                if (!var_885) {
                    var_890 = wp::extract(var_74, var_889);
                    var_891 = wp::sub(var_888, var_890);
                }
                var_892 = wp::where(var_885, var_887, var_891);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 370>
                var_894 = wp::bit_and(var_882, var_893);
                if (var_894) {
                    var_896 = wp::extract(var_74, var_895);
                }
                if (!var_894) {
                    var_899 = wp::extract(var_74, var_898);
                    var_900 = wp::sub(var_897, var_899);
                }
                var_901 = wp::where(var_894, var_896, var_900);
                var_902 = wp::mul(var_892, var_901);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 371>
                var_904 = wp::bit_and(var_882, var_903);
                if (var_904) {
                    var_906 = wp::extract(var_74, var_905);
                }
                if (!var_904) {
                    var_909 = wp::extract(var_74, var_908);
                    var_910 = wp::sub(var_907, var_909);
                }
                var_911 = wp::where(var_904, var_906, var_910);
                var_912 = wp::mul(var_902, var_911);
                // rx[j] = (                                                                  <L 368>
                wp::assign_inplace(var_4, var_882, var_912);
            }
            if (!var_883) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 374>
                var_914 = wp::bit_and(var_882, var_913);
                if (var_914) {
                }
                if (!var_914) {
                }
                var_917 = wp::where(var_914, var_915, var_916);
                var_919 = wp::bit_and(var_882, var_918);
                if (var_919) {
                    var_921 = wp::extract(var_74, var_920);
                }
                if (!var_919) {
                    var_924 = wp::extract(var_74, var_923);
                    var_925 = wp::sub(var_922, var_924);
                }
                var_926 = wp::where(var_919, var_921, var_925);
                var_927 = wp::mul(var_917, var_926);
                var_929 = wp::bit_and(var_882, var_928);
                if (var_929) {
                    var_931 = wp::extract(var_74, var_930);
                }
                if (!var_929) {
                    var_934 = wp::extract(var_74, var_933);
                    var_935 = wp::sub(var_932, var_934);
                }
                var_936 = wp::where(var_929, var_931, var_935);
                var_937 = wp::mul(var_927, var_936);
                wp::assign_inplace(var_4, var_882, var_937);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 375>
                var_939 = wp::bit_and(var_882, var_938);
                if (var_939) {
                    var_941 = wp::extract(var_74, var_940);
                }
                if (!var_939) {
                    var_944 = wp::extract(var_74, var_943);
                    var_945 = wp::sub(var_942, var_944);
                }
                var_946 = wp::where(var_939, var_941, var_945);
                var_948 = wp::bit_and(var_882, var_947);
                if (var_948) {
                }
                if (!var_948) {
                }
                var_951 = wp::where(var_948, var_949, var_950);
                var_952 = wp::mul(var_946, var_951);
                var_954 = wp::bit_and(var_882, var_953);
                if (var_954) {
                    var_956 = wp::extract(var_74, var_955);
                }
                if (!var_954) {
                    var_959 = wp::extract(var_74, var_958);
                    var_960 = wp::sub(var_957, var_959);
                }
                var_961 = wp::where(var_954, var_956, var_960);
                var_962 = wp::mul(var_952, var_961);
                wp::assign_inplace(var_6, var_882, var_962);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 376>
                var_964 = wp::bit_and(var_882, var_963);
                if (var_964) {
                    var_966 = wp::extract(var_74, var_965);
                }
                if (!var_964) {
                    var_969 = wp::extract(var_74, var_968);
                    var_970 = wp::sub(var_967, var_969);
                }
                var_971 = wp::where(var_964, var_966, var_970);
                var_973 = wp::bit_and(var_882, var_972);
                if (var_973) {
                    var_975 = wp::extract(var_74, var_974);
                }
                if (!var_973) {
                    var_978 = wp::extract(var_74, var_977);
                    var_979 = wp::sub(var_976, var_978);
                }
                var_980 = wp::where(var_973, var_975, var_979);
                var_981 = wp::mul(var_971, var_980);
                var_983 = wp::bit_and(var_882, var_982);
                if (var_983) {
                }
                if (!var_983) {
                }
                var_986 = wp::where(var_983, var_984, var_985);
                var_987 = wp::mul(var_981, var_986);
                wp::assign_inplace(var_8, var_882, var_987);
            }
            // return node, (rx, ry, rz)                                                      <L 377>
            var_988 = wp::tuple(var_4, var_6, var_8);
            ret_0 = var_14;
            ret_1 = var_988;
            return;
        }
        // x = 0 if coord[0] < 0.5 else 1                                                     <L 381>
        var_990 = wp::extract(var_74, var_989);
        var_992 = (var_990 < var_991);
        if (var_992) {
        }
        if (!var_992) {
        }
        var_995 = wp::where(var_992, var_993, var_994);
        // y = 0 if coord[1] < 0.5 else 1                                                     <L 382>
        var_997 = wp::extract(var_74, var_996);
        var_999 = (var_997 < var_998);
        if (var_999) {
        }
        if (!var_999) {
        }
        var_1002 = wp::where(var_999, var_1000, var_1001);
        // z = 0 if coord[2] < 0.5 else 1                                                     <L 383>
        var_1004 = wp::extract(var_74, var_1003);
        var_1006 = (var_1004 < var_1005);
        if (var_1006) {
        }
        if (!var_1006) {
        }
        var_1009 = wp::where(var_1006, var_1007, var_1008);
        // child = oct_child[node][4 * z + 2 * y + x]                                         <L 384>
        var_1010 = wp::address(var_oct_child, var_14);
        var_1012 = wp::mul(var_1011, var_1009);
        var_1014 = wp::mul(var_1013, var_1002);
        var_1015 = wp::add(var_1012, var_1014);
        var_1016 = wp::add(var_1015, var_995);
        var_1018 = wp::load(var_1010);
        var_1017 = wp::extract(var_1018, var_1016);
        // stack = child + root if child != -1 else -1                                        <L 385>
        var_1020 = (var_1017 != var_1019);
        if (var_1020) {
            var_1021 = wp::add(var_1017, var_root);
        }
        if (!var_1020) {
        }
        var_1023 = wp::where(var_1020, var_1021, var_1022);
        wp::assign(var_0, var_1023);
        wp::assign(var_2, var_71);
    goto start_while_0;
    end_while_0:;
    // wp.print("ERROR: Node not found\n")                                                    <L 387>
    wp::print(var_1024);
    // return -1, (rx, ry, rz)                                                                <L 388>
    var_1026 = wp::tuple(var_4, var_6, var_8);
    ret_0 = var_1025;
    ret_1 = var_1026;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:428
static CUDA_CALLABLE wp::float32 sample_volume_sdf_0(
    wp::vec_t<3, wp::float32> var_xyz,
    VolumeData_53ac1a2d var_volume_data)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::array_t<wp::vec_t<8, wp::int32>>* var_6;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_7;
    const bool var_8 = false;
    wp::int32* var_9;
    wp::int32 var_10;
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> var_11;
    wp::array_t<wp::vec_t<8, wp::int32>> var_12;
    wp::array_t<wp::vec_t<3, wp::float32>> var_13;
    wp::int32 var_14;
    const wp::int32 var_15 = 0;
    wp::vec_t<8, wp::float32> var_16;
    wp::array_t<wp::vec_t<8, wp::float32>>* var_17;
    wp::vec_t<8, wp::float32>* var_18;
    wp::array_t<wp::vec_t<8, wp::float32>> var_19;
    wp::float32 var_20;
    wp::vec_t<8, wp::float32> var_21;
    wp::float32 var_22;
    //---------
    // forward
    // def sample_volume_sdf(xyz: wp.vec3, volume_data: VolumeData) -> float:                 <L 429>
    // dist0, point = box_project(volume_data.center, volume_data.half_size, xyz)             <L 430>
    var_0 = &((var_volume_data).center);
    var_1 = &((var_volume_data).half_size);
    var_4 = wp::load(var_0);
    var_5 = wp::load(var_1);
    box_project_0(var_4, var_5, var_xyz, var_2, var_3);
    // node, weights = find_oct(volume_data.oct_child, volume_data.oct_aabb, point, grad=False, root=volume_data.root)       <L 431>
    var_6 = &((var_volume_data).oct_child);
    var_7 = &((var_volume_data).oct_aabb);
    var_9 = &((var_volume_data).root);
    var_12 = wp::load(var_6);
    var_13 = wp::load(var_7);
    var_14 = wp::load(var_9);
    find_oct_0(var_12, var_13, var_3, var_8, var_14, var_10, var_11);
    // return dist0 + wp.dot(weights[0], volume_data.oct_coeff[node])                         <L 432>
    var_16 = wp::extract<0>(var_11);
    var_17 = &((var_volume_data).oct_coeff);
    var_19 = wp::load(var_17);
    var_18 = wp::address(var_19, var_10);
    var_21 = wp::load(var_18);
    var_20 = wp::dot(var_16, var_21);
    var_22 = wp::add(var_2, var_20);
    return var_22;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:298
static CUDA_CALLABLE wp::float32 user_sdf_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<128, wp::float32> var_attr,
    wp::int32 var_sdf_type)
{
    //---------
    // primal vars
    const wp::str var_0 = "ERROR: user_sdf function must be implemented by user code\n";
    const wp::float32 var_1 = 0.0;
    //---------
    // forward
    // def user_sdf(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> float:                <L 299>
    // wp.printf("ERROR: user_sdf function must be implemented by user code\n")               <L 304>
    printf(var_0);
    // return 0.0                                                                             <L 305>
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:455
static CUDA_CALLABLE wp::float32 sdf_0(
    wp::int32 var_type,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<128, wp::float32> var_attr,
    wp::int32 var_sdf_type,
    VolumeData_53ac1a2d var_volume_data,
    MeshData_52eaa0fa var_mesh_data)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    bool var_12;
    wp::float32 var_13;
    const wp::int32 var_14 = 3;
    bool var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 5;
    bool var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 6;
    bool var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    bool var_24;
    wp::float32 var_25;
    bool var_26;
    const wp::int32 var_27 = 7;
    bool var_28;
    bool* var_29;
    bool var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::int32* var_33;
    wp::array_t<wp::int32>* var_34;
    wp::array_t<wp::int32>* var_35;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_36;
    wp::array_t<wp::vec_t<3, wp::int32>>* var_37;
    wp::int32* var_38;
    wp::vec_t<3, wp::float32>* var_39;
    wp::mat_t<3, 3, wp::float32>* var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::vec_t<3, wp::float32>* var_43;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::int32 var_46;
    wp::array_t<wp::int32> var_47;
    wp::array_t<wp::int32> var_48;
    wp::array_t<wp::vec_t<3, wp::float32>> var_49;
    wp::array_t<wp::vec_t<3, wp::int32>> var_50;
    wp::int32 var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::mat_t<3, 3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    bool var_58;
    wp::int32* var_59;
    wp::array_t<wp::int32>* var_60;
    wp::array_t<wp::int32>* var_61;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_62;
    wp::array_t<wp::vec_t<3, wp::int32>>* var_63;
    wp::int32* var_64;
    wp::vec_t<3, wp::float32>* var_65;
    wp::mat_t<3, 3, wp::float32>* var_66;
    wp::vec_t<3, wp::float32>* var_67;
    wp::vec_t<3, wp::float32>* var_68;
    wp::vec_t<3, wp::float32>* var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::float32 var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::int32 var_74;
    wp::array_t<wp::int32> var_75;
    wp::array_t<wp::int32> var_76;
    wp::array_t<wp::vec_t<3, wp::float32>> var_77;
    wp::array_t<wp::vec_t<3, wp::int32>> var_78;
    wp::int32 var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::mat_t<3, 3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32> var_86;
    const wp::int32 var_87 = 8;
    bool var_88;
    const wp::int32 var_89 = -1;
    bool var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    bool var_93;
    const wp::int32 var_94 = 7;
    bool var_95;
    bool* var_96;
    bool var_97;
    wp::float32 var_98;
    const wp::str var_99 = "ERROR: SDF type not implemented\n";
    const wp::float32 var_100 = 0.0;
    //---------
    // forward
    // def sdf(type: int, p: wp.vec3, attr: vec_pluginattr, sdf_type: int, volume_data: VolumeData, mesh_data: MeshData) -> float:       <L 456>
    // attr_vec3 = wp.vec3(attr[0], attr[1], attr[2])                                         <L 458>
    var_1 = wp::extract(var_attr, var_0);
    var_3 = wp::extract(var_attr, var_2);
    var_5 = wp::extract(var_attr, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // if type == GeomType.PLANE:                                                             <L 459>
    var_8 = (var_type == var_7);
    if (var_8) {
        // return p[2]                                                                        <L 460>
        var_10 = wp::extract(var_p, var_9);
        return var_10;
    }
    if (!var_8) {
        // elif type == GeomType.SPHERE:                                                      <L 461>
        var_12 = (var_type == var_11);
        if (var_12) {
            // return sphere(p, attr_vec3)                                                    <L 462>
            var_13 = sphere_0(var_p, var_6);
            return var_13;
        }
        if (!var_12) {
            // elif type == GeomType.CAPSULE:                                                 <L 463>
            var_15 = (var_type == var_14);
            if (var_15) {
                // return capsule(p, attr_vec3)                                               <L 464>
                var_16 = capsule_0(var_p, var_6);
                return var_16;
            }
            if (!var_15) {
                // elif type == GeomType.CYLINDER:                                            <L 465>
                var_18 = (var_type == var_17);
                if (var_18) {
                    // return cylinder(p, attr_vec3)                                          <L 466>
                    var_19 = cylinder_0(var_p, var_6);
                    return var_19;
                }
                if (!var_18) {
                    // elif type == GeomType.BOX:                                             <L 467>
                    var_21 = (var_type == var_20);
                    if (var_21) {
                        // return box(p, attr_vec3)                                           <L 468>
                        var_22 = box_0(var_p, var_6);
                        return var_22;
                    }
                    if (!var_21) {
                        // elif type == GeomType.ELLIPSOID:                                   <L 469>
                        var_24 = (var_type == var_23);
                        if (var_24) {
                            // return ellipsoid(p, attr_vec3)                                 <L 470>
                            var_25 = ellipsoid_0(var_p, var_6);
                            return var_25;
                        }
                        if (!var_24) {
                            // elif type == GeomType.MESH and mesh_data.valid:                <L 471>
                            var_28 = (var_type == var_27);
                            var_26 = var_28;
                            if (var_26) {
                                var_29 = &((var_mesh_data).valid);
                                var_30 = wp::load(var_29);
                                var_26 = var_26 && var_30;
                            }
                            if (var_26) {
                                // mesh_data.pnt = p                                          <L 472>
                                var_mesh_data.pnt = var_p;
                                // mesh_data.vec = -wp.normalize(p)                           <L 473>
                                var_31 = wp::normalize(var_p);
                                var_32 = wp::neg(var_31);
                                var_mesh_data.vec = var_32;
                                // dist, normal = ray_mesh(                                   <L 474>
                                // mesh_data.nmeshface,                                       <L 475>
                                var_33 = &((var_mesh_data).nmeshface);
                                // mesh_data.mesh_vertadr,                                    <L 476>
                                var_34 = &((var_mesh_data).mesh_vertadr);
                                // mesh_data.mesh_faceadr,                                    <L 477>
                                var_35 = &((var_mesh_data).mesh_faceadr);
                                // mesh_data.mesh_vert,                                       <L 478>
                                var_36 = &((var_mesh_data).mesh_vert);
                                // mesh_data.mesh_face,                                       <L 479>
                                var_37 = &((var_mesh_data).mesh_face);
                                // mesh_data.data_id,                                         <L 480>
                                var_38 = &((var_mesh_data).data_id);
                                // mesh_data.pos,                                             <L 481>
                                var_39 = &((var_mesh_data).pos);
                                // mesh_data.mat,                                             <L 482>
                                var_40 = &((var_mesh_data).mat);
                                // mesh_data.size,                                            <L 483>
                                var_41 = &((var_mesh_data).size);
                                // mesh_data.pnt,                                             <L 484>
                                var_42 = &((var_mesh_data).pnt);
                                // mesh_data.vec,                                             <L 485>
                                var_43 = &((var_mesh_data).vec);
                                var_46 = wp::load(var_33);
                                var_47 = wp::load(var_34);
                                var_48 = wp::load(var_35);
                                var_49 = wp::load(var_36);
                                var_50 = wp::load(var_37);
                                var_51 = wp::load(var_38);
                                var_52 = wp::load(var_39);
                                var_53 = wp::load(var_40);
                                var_54 = wp::load(var_41);
                                var_55 = wp::load(var_42);
                                var_56 = wp::load(var_43);
                                ray_mesh_0(var_46, var_47, var_48, var_49, var_50, var_51, var_52, var_53, var_54, var_55, var_56, var_44, var_45);
                                // if dist > wp.norm_l2(p):                                   <L 487>
                                var_57 = norm_l2_0(var_p);
                                var_58 = (var_44 > var_57);
                                if (var_58) {
                                    // dist, normal = ray_mesh(                               <L 488>
                                    // mesh_data.nmeshface,                                   <L 489>
                                    var_59 = &((var_mesh_data).nmeshface);
                                    // mesh_data.mesh_vertadr,                                <L 490>
                                    var_60 = &((var_mesh_data).mesh_vertadr);
                                    // mesh_data.mesh_faceadr,                                <L 491>
                                    var_61 = &((var_mesh_data).mesh_faceadr);
                                    // mesh_data.mesh_vert,                                   <L 492>
                                    var_62 = &((var_mesh_data).mesh_vert);
                                    // mesh_data.mesh_face,                                   <L 493>
                                    var_63 = &((var_mesh_data).mesh_face);
                                    // mesh_data.data_id,                                     <L 494>
                                    var_64 = &((var_mesh_data).data_id);
                                    // mesh_data.pos,                                         <L 495>
                                    var_65 = &((var_mesh_data).pos);
                                    // mesh_data.mat,                                         <L 496>
                                    var_66 = &((var_mesh_data).mat);
                                    // mesh_data.size,                                        <L 497>
                                    var_67 = &((var_mesh_data).size);
                                    // mesh_data.pnt,                                         <L 498>
                                    var_68 = &((var_mesh_data).pnt);
                                    // -mesh_data.vec,                                        <L 499>
                                    var_69 = &((var_mesh_data).vec);
                                    var_71 = wp::load(var_69);
                                    var_70 = wp::neg(var_71);
                                    var_74 = wp::load(var_59);
                                    var_75 = wp::load(var_60);
                                    var_76 = wp::load(var_61);
                                    var_77 = wp::load(var_62);
                                    var_78 = wp::load(var_63);
                                    var_79 = wp::load(var_64);
                                    var_80 = wp::load(var_65);
                                    var_81 = wp::load(var_66);
                                    var_82 = wp::load(var_67);
                                    var_83 = wp::load(var_68);
                                    ray_mesh_0(var_74, var_75, var_76, var_77, var_78, var_79, var_80, var_81, var_82, var_83, var_70, var_72, var_73);
                                    // return -dist                                           <L 501>
                                    var_84 = wp::neg(var_72);
                                    return var_84;
                                }
                                var_85 = wp::where(var_58, var_72, var_44);
                                var_86 = wp::where(var_58, var_73, var_45);
                                // return dist                                                <L 502>
                                return var_85;
                            }
                            if (!var_26) {
                                // elif type == GeomType.SDF:                                 <L 503>
                                var_88 = (var_type == var_87);
                                if (var_88) {
                                    // if sdf_type == -1:                                     <L 504>
                                    var_90 = (var_sdf_type == var_89);
                                    if (var_90) {
                                        // return sample_volume_sdf(p, volume_data)           <L 505>
                                        var_91 = sample_volume_sdf_0(var_p, var_volume_data);
                                        return var_91;
                                    }
                                    if (!var_90) {
                                        // return user_sdf(p, attr, sdf_type)                 <L 507>
                                        var_92 = user_sdf_0(var_p, var_attr, var_sdf_type);
                                        return var_92;
                                    }
                                }
                                if (!var_88) {
                                    // elif type == GeomType.MESH and volume_data.valid:       <L 508>
                                    var_95 = (var_type == var_94);
                                    var_93 = var_95;
                                    if (var_93) {
                                        var_96 = &((var_volume_data).valid);
                                        var_97 = wp::load(var_96);
                                        var_93 = var_93 && var_97;
                                    }
                                    if (var_93) {
                                        // return sample_volume_sdf(p, volume_data)           <L 509>
                                        var_98 = sample_volume_sdf_0(var_p, var_volume_data);
                                        return var_98;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // wp.printf("ERROR: SDF type not implemented\n")                                         <L 510>
    printf(var_99);
    // return 0.0                                                                             <L 511>
    return var_100;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:2138
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _transform_spatial_0(
    wp::vec_t<6, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> var_dif)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def _transform_spatial(vec: wp.spatial_vector, dif: wp.vec3) -> wp.vec3:               <L 2139>
    // return wp.spatial_bottom(vec) - wp.cross(dif, wp.spatial_top(vec))                     <L 2140>
    var_0 = wp::spatial_bottom(var_vec);
    var_1 = wp::spatial_top(var_vec);
    var_2 = wp::cross(var_dif, var_1);
    var_3 = wp::sub(var_0, var_2);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/support.py:325
static CUDA_CALLABLE wp::vec_t<6, wp::float32> _decode_pyramid_0(
    wp::int32 var_njmax_in,
    wp::array_t<wp::float32> var_pyramid,
    wp::int32 var_efc_address,
    wp::vec_t<5, wp::float32> var_mu,
    wp::int32 var_condim)
{
    //---------
    // primal vars
    wp::vec_t<6, wp::float32> var_0;
    const wp::int32 var_1 = 1;
    bool var_2;
    wp::float32* var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    const wp::int32 var_13 = 2;
    wp::int32 var_14;
    wp::int32 var_15;
    bool var_16;
    wp::float32* var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 0.0;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    bool var_24;
    const wp::int32 var_25 = 1;
    wp::int32 var_26;
    wp::float32* var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 0.0;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::int32 var_38;
    //---------
    // forward
    // def _decode_pyramid(njmax_in: int, pyramid: wp.array[float], efc_address: int, mu: vec5, condim: int) -> wp.spatial_vector:       <L 326>
    // force = wp.spatial_vector()                                                            <L 328>
    var_0 = wp::vec_t<6, wp::float32>();
    // if condim == 1:                                                                        <L 330>
    var_2 = (var_condim == var_1);
    if (var_2) {
        // force[0] = pyramid[efc_address]                                                    <L 331>
        var_3 = wp::address(var_pyramid, var_efc_address);
        var_5 = wp::load(var_3);
        wp::assign_inplace(var_0, var_4, var_5);
        // return force                                                                       <L 332>
        return var_0;
    }
    // force[0] = float(0.0)                                                                  <L 334>
    var_7 = wp::float(var_6);
    wp::assign_inplace(var_0, var_8, var_7);
    // for i in range(condim - 1):                                                            <L 335>
    var_10 = wp::sub(var_condim, var_9);
    var_11 = wp::range(var_10);
    start_for_1:;
        if (iter_cmp(var_11) == 0) goto end_for_1;
        var_12 = wp::iter_next(var_11);
        // adr = 2 * i + efc_address                                                          <L 336>
        var_14 = wp::mul(var_13, var_12);
        var_15 = wp::add(var_14, var_efc_address);
        // if adr < njmax_in:                                                                 <L 337>
        var_16 = (var_15 < var_njmax_in);
        if (var_16) {
            // dir1 = pyramid[adr]                                                            <L 338>
            var_17 = wp::address(var_pyramid, var_15);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        if (!var_16) {
            // dir1 = 0.0                                                                     <L 340>
        }
        var_21 = wp::where(var_16, var_18, var_20);
        // if adr + 1 < njmax_in:                                                             <L 341>
        var_23 = wp::add(var_15, var_22);
        var_24 = (var_23 < var_njmax_in);
        if (var_24) {
            // dir2 = pyramid[adr + 1]                                                        <L 342>
            var_26 = wp::add(var_15, var_25);
            var_27 = wp::address(var_pyramid, var_26);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
        }
        if (!var_24) {
            // dir2 = 0.0                                                                     <L 344>
        }
        var_31 = wp::where(var_24, var_28, var_30);
        // force[0] += dir1 + dir2                                                            <L 345>
        var_32 = wp::add(var_21, var_31);
        wp::add_inplace(var_0, var_33, var_32);
        // force[i + 1] = (dir1 - dir2) * mu[i]                                               <L 346>
        var_34 = wp::sub(var_21, var_31);
        var_35 = wp::extract(var_mu, var_12);
        var_36 = wp::mul(var_34, var_35);
        var_38 = wp::add(var_12, var_37);
        wp::assign_inplace(var_0, var_38, var_36);
        goto start_for_1;
    end_for_1:;
    // return force                                                                           <L 348>
    return var_0;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/support.py:351
static CUDA_CALLABLE wp::vec_t<6, wp::float32> contact_force_fn_0(
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::int32 var_worldid,
    wp::int32 var_contact_id,
    bool var_to_world_frame)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<6, wp::float32> var_6;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    wp::int32* var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    bool var_14;
    const wp::int32 var_15 = 0;
    bool var_16;
    const wp::int32 var_17 = 0;
    wp::int32* var_18;
    bool var_19;
    wp::int32 var_20;
    const wp::int32 var_21 = 0;
    bool var_22;
    const wp::int32 var_23 = 0;
    bool var_24;
    wp::array_t<wp::float32> var_25;
    wp::vec_t<5, wp::float32>* var_26;
    wp::vec_t<6, wp::float32> var_27;
    wp::vec_t<5, wp::float32> var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::int32* var_31;
    bool var_32;
    wp::int32 var_33;
    wp::int32* var_34;
    wp::float32* var_35;
    wp::int32 var_36;
    wp::float32 var_37;
    wp::vec_t<6, wp::float32> var_38;
    wp::vec_t<6, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::mat_t<3, 3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::mat_t<3, 3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32>* var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::mat_t<3, 3, wp::float32> var_47;
    wp::vec_t<6, wp::float32> var_48;
    wp::vec_t<6, wp::float32> var_49;
    //---------
    // forward
    // def contact_force_fn(                                                                  <L 352>
    // force = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                                <L 369>
    var_6 = wp::vec_t<6, wp::float32>({var_0, var_1, var_2, var_3, var_4, var_5});
    // condim = contact_dim_in[contact_id]                                                    <L 370>
    var_7 = wp::address(var_contact_dim_in, var_contact_id);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // efc_address = contact_efc_address_in[contact_id, 0]                                    <L 371>
    var_11 = wp::address(var_contact_efc_address_in, var_contact_id, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // if contact_id >= 0 and contact_id <= nacon_in[0] and efc_address >= 0:                 <L 373>
    var_16 = (var_contact_id >= var_15);
    var_14 = var_16;
    if (var_14) {
        var_18 = wp::address(var_nacon_in, var_17);
        var_20 = wp::load(var_18);
        var_19 = (var_contact_id <= var_20);
        var_14 = var_14 && var_19;
    }
    if (var_14) {
        var_22 = (var_12 >= var_21);
        var_14 = var_14 && var_22;
    }
    if (var_14) {
        // if opt_cone == ConeType.PYRAMIDAL:                                                 <L 374>
        var_24 = (var_opt_cone == var_23);
        if (var_24) {
            // force = _decode_pyramid(                                                       <L 375>
            // njmax_in,                                                                      <L 376>
            // efc_force_in[worldid],                                                         <L 377>
            var_25 = wp::view(var_efc_force_in, var_worldid);
            // efc_address,                                                                   <L 378>
            // contact_friction_in[contact_id],                                               <L 379>
            var_26 = wp::address(var_contact_friction_in, var_contact_id);
            // condim,                                                                        <L 380>
            var_28 = wp::load(var_26);
            var_27 = _decode_pyramid_0(var_njmax_in, var_25, var_12, var_28, var_8);
        }
        if (!var_24) {
            // for i in range(condim):                                                        <L 383>
            var_29 = wp::range(var_8);
            start_for_0:;
                if (iter_cmp(var_29) == 0) goto end_for_0;
                var_30 = wp::iter_next(var_29);
                // if contact_efc_address_in[contact_id, i] < njmax_in:                       <L 384>
                var_31 = wp::address(var_contact_efc_address_in, var_contact_id, var_30);
                var_33 = wp::load(var_31);
                var_32 = (var_33 < var_njmax_in);
                if (var_32) {
                    // force[i] = efc_force_in[worldid, contact_efc_address_in[contact_id, i]]       <L 385>
                    var_34 = wp::address(var_contact_efc_address_in, var_contact_id, var_30);
                    var_36 = wp::load(var_34);
                    var_35 = wp::address(var_efc_force_in, var_worldid, var_36);
                    var_37 = wp::load(var_35);
                    wp::assign_inplace(var_6, var_30, var_37);
                }
                goto start_for_0;
            end_for_0:;
        }
        var_38 = wp::where(var_24, var_27, var_6);
    }
    var_39 = wp::where(var_14, var_38, var_6);
    // if to_world_frame:                                                                     <L 387>
    if (var_to_world_frame) {
        // t = wp.spatial_top(force) @ contact_frame_in[contact_id]                           <L 389>
        var_40 = wp::spatial_top(var_39);
        var_41 = wp::address(var_contact_frame_in, var_contact_id);
        var_43 = wp::load(var_41);
        var_42 = wp::mul(var_40, var_43);
        // b = wp.spatial_bottom(force) @ contact_frame_in[contact_id]                        <L 390>
        var_44 = wp::spatial_bottom(var_39);
        var_45 = wp::address(var_contact_frame_in, var_contact_id);
        var_47 = wp::load(var_45);
        var_46 = wp::mul(var_44, var_47);
        // force = wp.spatial_vector(t, b)                                                    <L 391>
        var_48 = wp::vec_t<6, wp::float32>(var_42, var_46);
    }
    var_49 = wp::where(var_to_world_frame, var_48, var_39);
    // return force                                                                           <L 393>
    return var_49;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1509
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _accelerometer_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::mat_t<3, 3, wp::float32>* var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32> var_5;
    wp::mat_t<3, 3, wp::float32> var_6;
    wp::vec_t<6, wp::float32>* var_7;
    wp::vec_t<6, wp::float32> var_8;
    wp::vec_t<6, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<6, wp::float32>* var_12;
    wp::vec_t<6, wp::float32> var_13;
    wp::vec_t<6, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::int32* var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::int32 var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    //---------
    // forward
    // def _accelerometer(                                                                    <L 1510>
    // bodyid = site_bodyid[objid]                                                            <L 1524>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // rot = site_xmat_in[worldid, objid]                                                     <L 1525>
    var_3 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // rotT = wp.transpose(rot)                                                               <L 1526>
    var_6 = wp::transpose(var_4);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 1527>
    var_7 = wp::address(var_cvel_in, var_worldid, var_1);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // cvel_top = wp.spatial_top(cvel)                                                        <L 1528>
    var_10 = wp::spatial_top(var_8);
    // cvel_bottom = wp.spatial_bottom(cvel)                                                  <L 1529>
    var_11 = wp::spatial_bottom(var_8);
    // cacc = cacc_in[worldid, bodyid]                                                        <L 1530>
    var_12 = wp::address(var_cacc_in, var_worldid, var_1);
    var_14 = wp::load(var_12);
    var_13 = wp::copy(var_14);
    // cacc_top = wp.spatial_top(cacc)                                                        <L 1531>
    var_15 = wp::spatial_top(var_13);
    // cacc_bottom = wp.spatial_bottom(cacc)                                                  <L 1532>
    var_16 = wp::spatial_bottom(var_13);
    // dif = site_xpos_in[worldid, objid] - subtree_com_in[worldid, body_rootid[bodyid]]       <L 1533>
    var_17 = wp::address(var_site_xpos_in, var_worldid, var_objid);
    var_18 = wp::address(var_body_rootid, var_1);
    var_20 = wp::load(var_18);
    var_19 = wp::address(var_subtree_com_in, var_worldid, var_20);
    var_22 = wp::load(var_17);
    var_23 = wp::load(var_19);
    var_21 = wp::sub(var_22, var_23);
    // ang = rotT @ cvel_top                                                                  <L 1534>
    var_24 = wp::mul(var_6, var_10);
    // lin = rotT @ (cvel_bottom - wp.cross(dif, cvel_top))                                   <L 1535>
    var_25 = wp::cross(var_21, var_10);
    var_26 = wp::sub(var_11, var_25);
    var_27 = wp::mul(var_6, var_26);
    // acc = rotT @ (cacc_bottom - wp.cross(dif, cacc_top))                                   <L 1536>
    var_28 = wp::cross(var_21, var_15);
    var_29 = wp::sub(var_16, var_28);
    var_30 = wp::mul(var_6, var_29);
    // correction = wp.cross(ang, lin)                                                        <L 1537>
    var_31 = wp::cross(var_24, var_27);
    // return acc + correction                                                                <L 1538>
    var_32 = wp::add(var_30, var_31);
    return var_32;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<3, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::int32* var_9;
    const wp::int32 var_10 = 41;
    const wp::int32 var_11 = 41;
    wp::int32 var_12;
    bool var_13;
    wp::int32 var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    bool var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::range_t var_35;
    wp::int32 var_36;
    wp::float32 var_37;
    wp::int32 var_38;
    //---------
    // forward
    // def _write_vector(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 14>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 15>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 17>
    var_8 = (var_4 > var_7);
    var_6 = var_8;
    if (var_6) {
        var_9 = wp::address(var_sensor_type, var_sensorid);
        var_12 = wp::int(var_11);
        var_14 = wp::load(var_9);
        var_13 = (var_14 == var_12);
        var_15 = wp::unot(var_13);
        var_6 = var_6 && var_15;
    }
    if (var_6) {
        // datatype = sensor_datatype[sensorid]                                               <L 18>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 19>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // for i in range(sensordim):                                                     <L 20>
            var_21 = wp::range(var_sensordim);
            start_for_0:;
                if (iter_cmp(var_21) == 0) goto end_for_0;
                var_22 = wp::iter_next(var_21);
                // out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)                        <L 21>
                var_23 = wp::extract(var_sensor, var_22);
                var_24 = wp::neg(var_4);
                var_25 = wp::clamp(var_23, var_24, var_4);
                var_26 = wp::add(var_1, var_22);
                wp::array_store(var_out, var_26, var_25);
                goto start_for_0;
            end_for_0:;
            // return                                                                         <L 22>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 23>
            var_28 = (var_17 == var_27);
            if (var_28) {
                // for i in range(sensordim):                                                 <L 24>
                var_29 = wp::range(var_sensordim);
                start_for_3:;
                    if (iter_cmp(var_29) == 0) goto end_for_3;
                    var_30 = wp::iter_next(var_29);
                    // out[adr + i] = wp.min(sensor[i], cutoff)                               <L 25>
                    var_31 = wp::extract(var_sensor, var_30);
                    var_32 = wp::min(var_31, var_4);
                    var_33 = wp::add(var_1, var_30);
                    wp::array_store(var_out, var_33, var_32);
                    goto start_for_3;
                end_for_3:;
                // return                                                                     <L 26>
                return;
            }
        }
        var_34 = wp::where(var_20, var_22, var_30);
    }
    // for i in range(sensordim):                                                             <L 28>
    var_35 = wp::range(var_sensordim);
    start_for_6:;
        if (iter_cmp(var_35) == 0) goto end_for_6;
        var_36 = wp::iter_next(var_35);
        // out[adr + i] = sensor[i]                                                           <L 29>
        var_37 = wp::extract(var_sensor, var_36);
        var_38 = wp::add(var_1, var_36);
        wp::array_store(var_out, var_38, var_37);
        goto start_for_6;
    end_for_6:;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1541
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _force_0(
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::vec_t<6, wp::float32>* var_3;
    wp::vec_t<6, wp::float32> var_4;
    wp::vec_t<6, wp::float32> var_5;
    wp::mat_t<3, 3, wp::float32>* var_6;
    wp::mat_t<3, 3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _force(                                                                            <L 1542>
    // bodyid = site_bodyid[objid]                                                            <L 1552>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cfrc_int = cfrc_int_in[worldid, bodyid]                                                <L 1553>
    var_3 = wp::address(var_cfrc_int_in, var_worldid, var_1);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // site_xmat = site_xmat_in[worldid, objid]                                               <L 1554>
    var_6 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // return wp.transpose(site_xmat) @ wp.spatial_bottom(cfrc_int)                           <L 1555>
    var_9 = wp::transpose(var_7);
    var_10 = wp::spatial_bottom(var_4);
    var_11 = wp::mul(var_9, var_10);
    return var_11;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1558
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _torque_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::vec_t<6, wp::float32>* var_3;
    wp::vec_t<6, wp::float32> var_4;
    wp::vec_t<6, wp::float32> var_5;
    wp::mat_t<3, 3, wp::float32>* var_6;
    wp::mat_t<3, 3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::int32* var_10;
    wp::vec_t<3, wp::float32>* var_11;
    wp::int32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    //---------
    // forward
    // def _torque(                                                                           <L 1559>
    // bodyid = site_bodyid[objid]                                                            <L 1572>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cfrc_int = cfrc_int_in[worldid, bodyid]                                                <L 1573>
    var_3 = wp::address(var_cfrc_int_in, var_worldid, var_1);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // site_xmat = site_xmat_in[worldid, objid]                                               <L 1574>
    var_6 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // dif = site_xpos_in[worldid, objid] - subtree_com_in[worldid, body_rootid[bodyid]]       <L 1575>
    var_9 = wp::address(var_site_xpos_in, var_worldid, var_objid);
    var_10 = wp::address(var_body_rootid, var_1);
    var_12 = wp::load(var_10);
    var_11 = wp::address(var_subtree_com_in, var_worldid, var_12);
    var_14 = wp::load(var_9);
    var_15 = wp::load(var_11);
    var_13 = wp::sub(var_14, var_15);
    // return wp.transpose(site_xmat) @ (wp.spatial_top(cfrc_int) - wp.cross(dif, wp.spatial_bottom(cfrc_int)))       <L 1576>
    var_16 = wp::transpose(var_7);
    var_17 = wp::spatial_top(var_4);
    var_18 = wp::spatial_bottom(var_4);
    var_19 = wp::cross(var_13, var_18);
    var_20 = wp::sub(var_17, var_19);
    var_21 = wp::mul(var_16, var_20);
    return var_21;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1579
static CUDA_CALLABLE wp::float32 _actuator_force_0(
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _actuator_force(actuator_force_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 1580>
    // return actuator_force_in[worldid, objid]                                               <L 1581>
    var_0 = wp::address(var_actuator_force_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1584
static CUDA_CALLABLE wp::float32 _joint_actuator_force_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::float32* var_1;
    wp::int32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def _joint_actuator_force(                                                             <L 1585>
    // return qfrc_actuator_in[worldid, jnt_dofadr[objid]]                                    <L 1594>
    var_0 = wp::address(var_jnt_dofadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::address(var_qfrc_actuator_in, var_worldid, var_2);
    var_4 = wp::load(var_1);
    var_3 = wp::copy(var_4);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1677
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _framelinacc_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::int32 var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    const wp::int32 var_6 = 2;
    bool var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 5;
    bool var_13;
    wp::int32* var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 6;
    bool var_21;
    wp::int32* var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    const wp::int32 var_28 = 7;
    bool var_29;
    wp::int32* var_30;
    wp::int32 var_31;
    wp::int32 var_32;
    wp::vec_t<3, wp::float32>* var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    const wp::int32 var_36 = 0;
    const wp::float32 var_37 = 0.0;
    wp::vec_t<3, wp::float32> var_38;
    wp::int32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::int32 var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::int32 var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::int32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<6, wp::float32>* var_49;
    wp::vec_t<6, wp::float32> var_50;
    wp::vec_t<6, wp::float32> var_51;
    wp::vec_t<6, wp::float32>* var_52;
    wp::vec_t<6, wp::float32> var_53;
    wp::vec_t<6, wp::float32> var_54;
    wp::int32* var_55;
    wp::vec_t<3, wp::float32>* var_56;
    wp::int32 var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    //---------
    // forward
    // def _framelinacc(                                                                      <L 1678>
    // if objtype == ObjType.BODY:                                                            <L 1698>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // bodyid = objid                                                                     <L 1699>
        var_2 = wp::copy(var_objid);
        // pos = xipos_in[worldid, objid]                                                     <L 1700>
        var_3 = wp::address(var_xipos_in, var_worldid, var_objid);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 1701>
        var_7 = (var_objtype == var_6);
        if (var_7) {
            // bodyid = objid                                                                 <L 1702>
            var_8 = wp::copy(var_objid);
            // pos = xpos_in[worldid, objid]                                                  <L 1703>
            var_9 = wp::address(var_xpos_in, var_worldid, var_objid);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        if (!var_7) {
            // elif objtype == ObjType.GEOM:                                                  <L 1704>
            var_13 = (var_objtype == var_12);
            if (var_13) {
                // bodyid = geom_bodyid[objid]                                                <L 1705>
                var_14 = wp::address(var_geom_bodyid, var_objid);
                var_16 = wp::load(var_14);
                var_15 = wp::copy(var_16);
                // pos = geom_xpos_in[worldid, objid]                                         <L 1706>
                var_17 = wp::address(var_geom_xpos_in, var_worldid, var_objid);
                var_19 = wp::load(var_17);
                var_18 = wp::copy(var_19);
            }
            if (!var_13) {
                // elif objtype == ObjType.SITE:                                              <L 1707>
                var_21 = (var_objtype == var_20);
                if (var_21) {
                    // bodyid = site_bodyid[objid]                                            <L 1708>
                    var_22 = wp::address(var_site_bodyid, var_objid);
                    var_24 = wp::load(var_22);
                    var_23 = wp::copy(var_24);
                    // pos = site_xpos_in[worldid, objid]                                     <L 1709>
                    var_25 = wp::address(var_site_xpos_in, var_worldid, var_objid);
                    var_27 = wp::load(var_25);
                    var_26 = wp::copy(var_27);
                }
                if (!var_21) {
                    // elif objtype == ObjType.CAMERA:                                        <L 1710>
                    var_29 = (var_objtype == var_28);
                    if (var_29) {
                        // bodyid = cam_bodyid[objid]                                         <L 1711>
                        var_30 = wp::address(var_cam_bodyid, var_objid);
                        var_32 = wp::load(var_30);
                        var_31 = wp::copy(var_32);
                        // pos = cam_xpos_in[worldid, objid]                                  <L 1712>
                        var_33 = wp::address(var_cam_xpos_in, var_worldid, var_objid);
                        var_35 = wp::load(var_33);
                        var_34 = wp::copy(var_35);
                    }
                    if (!var_29) {
                        // bodyid = 0                                                         <L 1714>
                        // pos = wp.vec3(0.0)                                                 <L 1715>
                        var_38 = wp::vec_t<3, wp::float32>(var_37);
                    }
                    var_39 = wp::where(var_29, var_31, var_36);
                    var_40 = wp::where(var_29, var_34, var_38);
                }
                var_41 = wp::where(var_21, var_23, var_39);
                var_42 = wp::where(var_21, var_26, var_40);
            }
            var_43 = wp::where(var_13, var_15, var_41);
            var_44 = wp::where(var_13, var_18, var_42);
        }
        var_45 = wp::where(var_7, var_8, var_43);
        var_46 = wp::where(var_7, var_10, var_44);
    }
    var_47 = wp::where(var_1, var_2, var_45);
    var_48 = wp::where(var_1, var_4, var_46);
    // cacc = cacc_in[worldid, bodyid]                                                        <L 1717>
    var_49 = wp::address(var_cacc_in, var_worldid, var_47);
    var_51 = wp::load(var_49);
    var_50 = wp::copy(var_51);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 1718>
    var_52 = wp::address(var_cvel_in, var_worldid, var_47);
    var_54 = wp::load(var_52);
    var_53 = wp::copy(var_54);
    // offset = pos - subtree_com_in[worldid, body_rootid[bodyid]]                            <L 1719>
    var_55 = wp::address(var_body_rootid, var_47);
    var_57 = wp::load(var_55);
    var_56 = wp::address(var_subtree_com_in, var_worldid, var_57);
    var_59 = wp::load(var_56);
    var_58 = wp::sub(var_48, var_59);
    // ang = wp.spatial_top(cvel)                                                             <L 1720>
    var_60 = wp::spatial_top(var_53);
    // lin = wp.spatial_bottom(cvel) - wp.cross(offset, ang)                                  <L 1721>
    var_61 = wp::spatial_bottom(var_53);
    var_62 = wp::cross(var_58, var_60);
    var_63 = wp::sub(var_61, var_62);
    // acc = wp.spatial_bottom(cacc) - wp.cross(offset, wp.spatial_top(cacc))                 <L 1722>
    var_64 = wp::spatial_bottom(var_50);
    var_65 = wp::spatial_top(var_50);
    var_66 = wp::cross(var_58, var_65);
    var_67 = wp::sub(var_64, var_66);
    // correction = wp.cross(ang, lin)                                                        <L 1723>
    var_68 = wp::cross(var_60, var_63);
    // return acc + correction                                                                <L 1725>
    var_69 = wp::add(var_67, var_68);
    return var_69;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1728
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frameangacc_0(
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 1;
    bool var_2;
    const wp::int32 var_3 = 2;
    bool var_4;
    wp::int32 var_5;
    const wp::int32 var_6 = 5;
    bool var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    const wp::int32 var_11 = 6;
    bool var_12;
    wp::int32* var_13;
    wp::int32 var_14;
    wp::int32 var_15;
    const wp::int32 var_16 = 7;
    bool var_17;
    wp::int32* var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    const wp::int32 var_21 = 0;
    wp::int32 var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    wp::int32 var_25;
    wp::vec_t<6, wp::float32>* var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<6, wp::float32> var_28;
    //---------
    // forward
    // def _frameangacc(                                                                      <L 1729>
    // if objtype == ObjType.BODY or objtype == ObjType.XBODY:                                <L 1741>
    var_2 = (var_objtype == var_1);
    var_0 = var_2;
    if (!var_0) {
        var_4 = (var_objtype == var_3);
        var_0 = var_0 || var_4;
    }
    if (var_0) {
        // bodyid = objid                                                                     <L 1742>
        var_5 = wp::copy(var_objid);
    }
    if (!var_0) {
        // elif objtype == ObjType.GEOM:                                                      <L 1743>
        var_7 = (var_objtype == var_6);
        if (var_7) {
            // bodyid = geom_bodyid[objid]                                                    <L 1744>
            var_8 = wp::address(var_geom_bodyid, var_objid);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_7) {
            // elif objtype == ObjType.SITE:                                                  <L 1745>
            var_12 = (var_objtype == var_11);
            if (var_12) {
                // bodyid = site_bodyid[objid]                                                <L 1746>
                var_13 = wp::address(var_site_bodyid, var_objid);
                var_15 = wp::load(var_13);
                var_14 = wp::copy(var_15);
            }
            if (!var_12) {
                // elif objtype == ObjType.CAMERA:                                            <L 1747>
                var_17 = (var_objtype == var_16);
                if (var_17) {
                    // bodyid = cam_bodyid[objid]                                             <L 1748>
                    var_18 = wp::address(var_cam_bodyid, var_objid);
                    var_20 = wp::load(var_18);
                    var_19 = wp::copy(var_20);
                }
                if (!var_17) {
                    // bodyid = 0                                                             <L 1750>
                }
                var_22 = wp::where(var_17, var_19, var_21);
            }
            var_23 = wp::where(var_12, var_14, var_22);
        }
        var_24 = wp::where(var_7, var_9, var_23);
    }
    var_25 = wp::where(var_0, var_5, var_24);
    // return wp.spatial_top(cacc_in[worldid, bodyid])                                        <L 1752>
    var_26 = wp::address(var_cacc_in, var_worldid, var_25);
    var_28 = wp::load(var_26);
    var_27 = wp::spatial_top(var_28);
    return var_27;
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:161
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::int32 var_4 = 3;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::float32 var_12 = 2.0;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 3.141592653589793;
    bool var_18;
    const wp::float32 var_19 = 2.0;
    const wp::float32 var_20 = 3.141592653589793;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    //---------
    // forward
    // def quat_to_vel(quat: wp.quat) -> wp.vec3:                                             <L 162>
    // axis = wp.vec3(quat[1], quat[2], quat[3])                                              <L 163>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // sin_a_2 = wp.norm_l2(axis)                                                             <L 164>
    var_7 = norm_l2_0(var_6);
    // if sin_a_2 == 0.0:                                                                     <L 166>
    var_9 = (var_7 == var_8);
    if (var_9) {
        // return wp.vec3(0.0)                                                                <L 167>
        var_11 = wp::vec_t<3, wp::float32>(var_10);
        return var_11;
    }
    // speed = 2.0 * wp.atan2(sin_a_2, quat[0])                                               <L 169>
    var_14 = wp::extract(var_quat, var_13);
    var_15 = wp::atan2(var_7, var_14);
    var_16 = wp::mul(var_12, var_15);
    // if speed > wp.pi:                                                                      <L 171>
    var_18 = (var_16 > var_17);
    if (var_18) {
        // speed -= 2.0 * wp.pi                                                               <L 172>
        var_21 = wp::mul(var_19, var_20);
        var_22 = wp::sub(var_16, var_21);
    }
    var_23 = wp::where(var_18, var_22, var_16);
    // return axis * speed / sin_a_2                                                          <L 174>
    var_24 = wp::mul(var_6, var_23);
    var_25 = wp::div(var_24, var_7);
    return var_25;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:177
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 3;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::quat_t<wp::float32> var_11;
    wp::quat_t<wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def quat_sub(qa: wp.quat, qb: wp.quat) -> wp.vec3:                                     <L 178>
    // qneg = wp.quat(qb[0], -qb[1], -qb[2], -qb[3])                                          <L 181>
    var_1 = wp::extract(var_qb, var_0);
    var_3 = wp::extract(var_qb, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_qb, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_qb, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    // qdif = mul_quat(qneg, qa)                                                              <L 182>
    var_12 = mul_quat_0(var_11, var_qa);
    // return quat_to_vel(qdif)                                                               <L 185>
    var_13 = quat_to_vel_0(var_12);
    return var_13;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:322
static CUDA_CALLABLE wp::int32 upper_tri_index_0(
    wp::int32 var_n,
    wp::int32 var_i,
    wp::int32 var_j)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 3;
    wp::int32 var_4;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    wp::int32 var_8;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    //---------
    // forward
    // def upper_tri_index(n: int, i: int, j: int) -> int:                                    <L 323>
    // return (i * (2 * n - i - 3)) // 2 + j - 1                                              <L 325>
    var_1 = wp::mul(var_0, var_n);
    var_2 = wp::sub(var_1, var_i);
    var_4 = wp::sub(var_2, var_3);
    var_5 = wp::mul(var_i, var_4);
    var_7 = wp::floordiv(var_5, var_6);
    var_8 = wp::add(var_7, var_j);
    var_10 = wp::sub(var_8, var_9);
    return var_10;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:675
static CUDA_CALLABLE bool inside_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_point)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 2;
    bool var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 3;
    bool var_13;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    const wp::int32 var_38 = 0;
    wp::float32 var_39;
    wp::float32 var_40;
    bool var_41;
    const wp::int32 var_42 = 4;
    bool var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::float32 var_45;
    const wp::float32 var_46 = 1.0;
    bool var_47;
    const wp::int32 var_48 = 5;
    bool var_49;
    bool var_50;
    const wp::int32 var_51 = 2;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::int32 var_54 = 1;
    wp::float32 var_55;
    bool var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    const wp::int32 var_70 = 0;
    wp::float32 var_71;
    wp::float32 var_72;
    bool var_73;
    const wp::int32 var_74 = 6;
    bool var_75;
    bool var_76;
    const wp::int32 var_77 = 0;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 0;
    wp::float32 var_81;
    bool var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 1;
    wp::float32 var_87;
    bool var_88;
    const wp::int32 var_89 = 2;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 2;
    wp::float32 var_93;
    bool var_94;
    const wp::int32 var_95 = 0;
    bool var_96;
    const wp::int32 var_97 = 2;
    wp::float32 var_98;
    const wp::float32 var_99 = 0.0;
    bool var_100;
    const bool var_101 = false;
    //---------
    // forward
    // def inside_geom(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, geomtype: int, point: wp.vec3) -> bool:       <L 676>
    // vec = point - pos                                                                      <L 679>
    var_0 = wp::sub(var_point, var_pos);
    // if geomtype == GeomType.SPHERE:                                                        <L 682>
    var_2 = (var_geomtype == var_1);
    if (var_2) {
        // return wp.dot(vec, vec) < size[0] * size[0]                                        <L 683>
        var_3 = wp::dot(var_0, var_0);
        var_5 = wp::extract(var_size, var_4);
        var_7 = wp::extract(var_size, var_6);
        var_8 = wp::mul(var_5, var_7);
        var_9 = (var_3 < var_8);
        return var_9;
    }
    // plocal = wp.transpose(mat) @ vec                                                       <L 686>
    var_10 = wp::transpose(var_mat);
    var_11 = wp::mul(var_10, var_0);
    // if geomtype == GeomType.CAPSULE:                                                       <L 689>
    var_13 = (var_geomtype == var_12);
    if (var_13) {
        // z = plocal[2]                                                                      <L 690>
        var_15 = wp::extract(var_11, var_14);
        // z_clamped = wp.clamp(z, -size[1], size[1])                                         <L 691>
        var_17 = wp::extract(var_size, var_16);
        var_18 = wp::neg(var_17);
        var_20 = wp::extract(var_size, var_19);
        var_21 = wp::clamp(var_15, var_18, var_20);
        // z_dif = z - z_clamped                                                              <L 692>
        var_22 = wp::sub(var_15, var_21);
        // z_dist_sq = z_dif * z_dif                                                          <L 693>
        var_23 = wp::mul(var_22, var_22);
        // return plocal[0] * plocal[0] + plocal[1] * plocal[1] + z_dist_sq < size[0] * size[0]       <L 694>
        var_25 = wp::extract(var_11, var_24);
        var_27 = wp::extract(var_11, var_26);
        var_28 = wp::mul(var_25, var_27);
        var_30 = wp::extract(var_11, var_29);
        var_32 = wp::extract(var_11, var_31);
        var_33 = wp::mul(var_30, var_32);
        var_34 = wp::add(var_28, var_33);
        var_35 = wp::add(var_34, var_23);
        var_37 = wp::extract(var_size, var_36);
        var_39 = wp::extract(var_size, var_38);
        var_40 = wp::mul(var_37, var_39);
        var_41 = (var_35 < var_40);
        return var_41;
    }
    if (!var_13) {
        // elif geomtype == GeomType.ELLIPSOID:                                               <L 695>
        var_43 = (var_geomtype == var_42);
        if (var_43) {
            // plocalsize = wp.cw_div(plocal, size)                                           <L 696>
            var_44 = wp::cw_div(var_11, var_size);
            // return wp.dot(plocalsize, plocalsize) < 1.0                                    <L 697>
            var_45 = wp::dot(var_44, var_44);
            var_47 = (var_45 < var_46);
            return var_47;
        }
        if (!var_43) {
            // elif geomtype == GeomType.CYLINDER:                                            <L 698>
            var_49 = (var_geomtype == var_48);
            if (var_49) {
                // return (wp.abs(plocal[2]) < size[1]) and (plocal[0] * plocal[0] + plocal[1] * plocal[1] < size[0] * size[0])       <L 699>
                var_52 = wp::extract(var_11, var_51);
                var_53 = wp::abs(var_52);
                var_55 = wp::extract(var_size, var_54);
                var_56 = (var_53 < var_55);
                var_50 = var_56;
                if (var_50) {
                    var_58 = wp::extract(var_11, var_57);
                    var_60 = wp::extract(var_11, var_59);
                    var_61 = wp::mul(var_58, var_60);
                    var_63 = wp::extract(var_11, var_62);
                    var_65 = wp::extract(var_11, var_64);
                    var_66 = wp::mul(var_63, var_65);
                    var_67 = wp::add(var_61, var_66);
                    var_69 = wp::extract(var_size, var_68);
                    var_71 = wp::extract(var_size, var_70);
                    var_72 = wp::mul(var_69, var_71);
                    var_73 = (var_67 < var_72);
                    var_50 = var_50 && var_73;
                }
                return var_50;
            }
            if (!var_49) {
                // elif geomtype == GeomType.BOX:                                             <L 700>
                var_75 = (var_geomtype == var_74);
                if (var_75) {
                    // return wp.abs(plocal[0]) < size[0] and wp.abs(plocal[1]) < size[1] and wp.abs(plocal[2]) < size[2]       <L 701>
                    var_78 = wp::extract(var_11, var_77);
                    var_79 = wp::abs(var_78);
                    var_81 = wp::extract(var_size, var_80);
                    var_82 = (var_79 < var_81);
                    var_76 = var_82;
                    if (var_76) {
                        var_84 = wp::extract(var_11, var_83);
                        var_85 = wp::abs(var_84);
                        var_87 = wp::extract(var_size, var_86);
                        var_88 = (var_85 < var_87);
                        var_76 = var_76 && var_88;
                    }
                    if (var_76) {
                        var_90 = wp::extract(var_11, var_89);
                        var_91 = wp::abs(var_90);
                        var_93 = wp::extract(var_size, var_92);
                        var_94 = (var_91 < var_93);
                        var_76 = var_76 && var_94;
                    }
                    return var_76;
                }
                if (!var_75) {
                    // elif geomtype == GeomType.PLANE:                                       <L 702>
                    var_96 = (var_geomtype == var_95);
                    if (var_96) {
                        // return plocal[2] < 0.0                                             <L 703>
                        var_98 = wp::extract(var_11, var_97);
                        var_100 = (var_98 < var_99);
                        return var_100;
                    }
                }
            }
        }
    }
    // return False                                                                           <L 705>
    return var_101;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:2311
static CUDA_CALLABLE bool _check_match_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::int32 var_body,
    wp::int32 var_geom,
    wp::int32 var_objtype,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const bool var_2 = true;
    const wp::int32 var_3 = 6;
    bool var_4;
    const bool var_5 = true;
    const wp::int32 var_6 = 5;
    bool var_7;
    bool var_8;
    const wp::int32 var_9 = 1;
    bool var_10;
    bool var_11;
    const wp::int32 var_12 = 2;
    bool var_13;
    bool var_14;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    bool var_18;
    const bool var_19 = false;
    //---------
    // forward
    // def _check_match(body_parentid: wp.array[int], body: int, geom: int, objtype: int, objid: int) -> bool:       <L 2312>
    // if objtype == ObjType.UNKNOWN:                                                         <L 2314>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // return True                                                                        <L 2315>
        return var_2;
    }
    // if objtype == ObjType.SITE:                                                            <L 2316>
    var_4 = (var_objtype == var_3);
    if (var_4) {
        // return True  # already passed site filter test                                     <L 2317>
        return var_5;
    }
    // if objtype == ObjType.GEOM:                                                            <L 2318>
    var_7 = (var_objtype == var_6);
    if (var_7) {
        // return objid == geom                                                               <L 2319>
        var_8 = (var_objid == var_geom);
        return var_8;
    }
    // if objtype == ObjType.BODY:                                                            <L 2320>
    var_10 = (var_objtype == var_9);
    if (var_10) {
        // return objid == body                                                               <L 2321>
        var_11 = (var_objid == var_body);
        return var_11;
    }
    // if objtype == ObjType.XBODY:                                                           <L 2322>
    var_13 = (var_objtype == var_12);
    if (var_13) {
        // while body > objid:                                                                <L 2324>
    start_while_4:;
        var_14 = (var_body > var_objid);
    if ((var_14) == false) goto end_while_4;
            // body = body_parentid[body]                                                     <L 2325>
            var_15 = wp::address(var_body_parentid, var_body);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            wp::assign(var_body, var_16);
    goto start_while_4;
    end_while_4:;
        // return body == objid                                                               <L 2326>
        var_18 = (var_body == var_objid);
        return var_18;
    }
    // return False                                                                           <L 2327>
    return var_19;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:963
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _velocimeter_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::mat_t<3, 3, wp::float32>* var_6;
    wp::mat_t<3, 3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::vec_t<6, wp::float32>* var_9;
    wp::vec_t<6, wp::float32> var_10;
    wp::vec_t<6, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::int32* var_14;
    wp::vec_t<3, wp::float32>* var_15;
    wp::int32 var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::mat_t<3, 3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    //---------
    // forward
    // def _velocimeter(                                                                      <L 964>
    // bodyid = site_bodyid[objid]                                                            <L 977>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // pos = site_xpos_in[worldid, objid]                                                     <L 978>
    var_3 = wp::address(var_site_xpos_in, var_worldid, var_objid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // rot = site_xmat_in[worldid, objid]                                                     <L 979>
    var_6 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 980>
    var_9 = wp::address(var_cvel_in, var_worldid, var_1);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // ang = wp.spatial_top(cvel)                                                             <L 981>
    var_12 = wp::spatial_top(var_10);
    // lin = wp.spatial_bottom(cvel)                                                          <L 982>
    var_13 = wp::spatial_bottom(var_10);
    // subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]                             <L 983>
    var_14 = wp::address(var_body_rootid, var_1);
    var_16 = wp::load(var_14);
    var_15 = wp::address(var_subtree_com_in, var_worldid, var_16);
    var_18 = wp::load(var_15);
    var_17 = wp::copy(var_18);
    // dif = pos - subtree_com                                                                <L 984>
    var_19 = wp::sub(var_4, var_17);
    // return wp.transpose(rot) @ (lin - wp.cross(dif, ang))                                  <L 985>
    var_20 = wp::transpose(var_7);
    var_21 = wp::cross(var_19, var_12);
    var_22 = wp::sub(var_13, var_21);
    var_23 = wp::mul(var_20, var_22);
    return var_23;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:988
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _gyro_0(
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::mat_t<3, 3, wp::float32>* var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32> var_5;
    wp::vec_t<6, wp::float32>* var_6;
    wp::vec_t<6, wp::float32> var_7;
    wp::vec_t<6, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _gyro(                                                                             <L 989>
    // bodyid = site_bodyid[objid]                                                            <L 999>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // rot = site_xmat_in[worldid, objid]                                                     <L 1000>
    var_3 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 1001>
    var_6 = wp::address(var_cvel_in, var_worldid, var_1);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // ang = wp.spatial_top(cvel)                                                             <L 1002>
    var_9 = wp::spatial_top(var_7);
    // return wp.transpose(rot) @ ang                                                         <L 1003>
    var_10 = wp::transpose(var_4);
    var_11 = wp::mul(var_10, var_9);
    return var_11;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1006
static CUDA_CALLABLE wp::float32 _joint_vel_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::float32* var_1;
    wp::int32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def _joint_vel(jnt_dofadr: wp.array[int], qvel_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 1007>
    // return qvel_in[worldid, jnt_dofadr[objid]]                                             <L 1008>
    var_0 = wp::address(var_jnt_dofadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::address(var_qvel_in, var_worldid, var_2);
    var_4 = wp::load(var_1);
    var_3 = wp::copy(var_4);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1011
static CUDA_CALLABLE wp::float32 _tendon_vel_0(
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _tendon_vel(ten_velocity_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 1012>
    // return ten_velocity_in[worldid, objid]                                                 <L 1013>
    var_0 = wp::address(var_ten_velocity_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1016
static CUDA_CALLABLE wp::float32 _actuator_vel_0(
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _actuator_vel(actuator_velocity_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 1017>
    // return actuator_velocity_in[worldid, objid]                                            <L 1018>
    var_0 = wp::address(var_actuator_velocity_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1021
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _ball_ang_vel_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::float32* var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::float32* var_8;
    const wp::int32 var_9 = 2;
    wp::int32 var_10;
    wp::float32* var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    //---------
    // forward
    // def _ball_ang_vel(jnt_dofadr: wp.array[int], qvel_in: wp.array2d[float], worldid: int, objid: int) -> wp.vec3:       <L 1022>
    // adr = jnt_dofadr[objid]                                                                <L 1023>
    var_0 = wp::address(var_jnt_dofadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // return wp.vec3(qvel_in[worldid, adr + 0], qvel_in[worldid, adr + 1], qvel_in[worldid, adr + 2])       <L 1024>
    var_4 = wp::add(var_1, var_3);
    var_5 = wp::address(var_qvel_in, var_worldid, var_4);
    var_7 = wp::add(var_1, var_6);
    var_8 = wp::address(var_qvel_in, var_worldid, var_7);
    var_10 = wp::add(var_1, var_9);
    var_11 = wp::address(var_qvel_in, var_worldid, var_10);
    var_13 = wp::load(var_5);
    var_14 = wp::load(var_8);
    var_15 = wp::load(var_11);
    var_12 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
    return var_12;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1065
static CUDA_CALLABLE void _cvel_offset_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::vec_t<6, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    bool var_7;
    wp::vec_t<3, wp::float32>* var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::int32 var_11;
    const wp::int32 var_12 = 5;
    bool var_13;
    wp::vec_t<3, wp::float32>* var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::int32* var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 6;
    bool var_21;
    wp::vec_t<3, wp::float32>* var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::int32* var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    const wp::int32 var_28 = 7;
    bool var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::int32* var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    const wp::float32 var_36 = 0.0;
    wp::vec_t<3, wp::float32> var_37;
    const wp::int32 var_38 = 0;
    wp::vec_t<3, wp::float32> var_39;
    wp::int32 var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::int32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::int32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::int32 var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::int32 var_48;
    wp::vec_t<6, wp::float32>* var_49;
    wp::int32* var_50;
    wp::vec_t<3, wp::float32>* var_51;
    wp::int32 var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<6, wp::float32> var_55;
    wp::vec_t<6, wp::float32> var_56;
    //---------
    // forward
    // def _cvel_offset(                                                                      <L 1066>
    // if objtype == ObjType.BODY:                                                            <L 1085>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // pos = xipos_in[worldid, objid]                                                     <L 1086>
        var_2 = wp::address(var_xipos_in, var_worldid, var_objid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // bodyid = objid                                                                     <L 1087>
        var_5 = wp::copy(var_objid);
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 1088>
        var_7 = (var_objtype == var_6);
        if (var_7) {
            // pos = xpos_in[worldid, objid]                                                  <L 1089>
            var_8 = wp::address(var_xpos_in, var_worldid, var_objid);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // bodyid = objid                                                                 <L 1090>
            var_11 = wp::copy(var_objid);
        }
        if (!var_7) {
            // elif objtype == ObjType.GEOM:                                                  <L 1091>
            var_13 = (var_objtype == var_12);
            if (var_13) {
                // pos = geom_xpos_in[worldid, objid]                                         <L 1092>
                var_14 = wp::address(var_geom_xpos_in, var_worldid, var_objid);
                var_16 = wp::load(var_14);
                var_15 = wp::copy(var_16);
                // bodyid = geom_bodyid[objid]                                                <L 1093>
                var_17 = wp::address(var_geom_bodyid, var_objid);
                var_19 = wp::load(var_17);
                var_18 = wp::copy(var_19);
            }
            if (!var_13) {
                // elif objtype == ObjType.SITE:                                              <L 1094>
                var_21 = (var_objtype == var_20);
                if (var_21) {
                    // pos = site_xpos_in[worldid, objid]                                     <L 1095>
                    var_22 = wp::address(var_site_xpos_in, var_worldid, var_objid);
                    var_24 = wp::load(var_22);
                    var_23 = wp::copy(var_24);
                    // bodyid = site_bodyid[objid]                                            <L 1096>
                    var_25 = wp::address(var_site_bodyid, var_objid);
                    var_27 = wp::load(var_25);
                    var_26 = wp::copy(var_27);
                }
                if (!var_21) {
                    // elif objtype == ObjType.CAMERA:                                        <L 1097>
                    var_29 = (var_objtype == var_28);
                    if (var_29) {
                        // pos = cam_xpos_in[worldid, objid]                                  <L 1098>
                        var_30 = wp::address(var_cam_xpos_in, var_worldid, var_objid);
                        var_32 = wp::load(var_30);
                        var_31 = wp::copy(var_32);
                        // bodyid = cam_bodyid[objid]                                         <L 1099>
                        var_33 = wp::address(var_cam_bodyid, var_objid);
                        var_35 = wp::load(var_33);
                        var_34 = wp::copy(var_35);
                    }
                    if (!var_29) {
                        // pos = wp.vec3(0.0)                                                 <L 1101>
                        var_37 = wp::vec_t<3, wp::float32>(var_36);
                        // bodyid = 0                                                         <L 1102>
                    }
                    var_39 = wp::where(var_29, var_31, var_37);
                    var_40 = wp::where(var_29, var_34, var_38);
                }
                var_41 = wp::where(var_21, var_23, var_39);
                var_42 = wp::where(var_21, var_26, var_40);
            }
            var_43 = wp::where(var_13, var_15, var_41);
            var_44 = wp::where(var_13, var_18, var_42);
        }
        var_45 = wp::where(var_7, var_9, var_43);
        var_46 = wp::where(var_7, var_11, var_44);
    }
    var_47 = wp::where(var_1, var_3, var_45);
    var_48 = wp::where(var_1, var_5, var_46);
    // return cvel_in[worldid, bodyid], pos - subtree_com_in[worldid, body_rootid[bodyid]]       <L 1104>
    var_49 = wp::address(var_cvel_in, var_worldid, var_48);
    var_50 = wp::address(var_body_rootid, var_48);
    var_52 = wp::load(var_50);
    var_51 = wp::address(var_subtree_com_in, var_worldid, var_52);
    var_54 = wp::load(var_51);
    var_53 = wp::sub(var_47, var_54);
    var_56 = wp::load(var_49);
    var_55 = wp::copy(var_56);
    ret_0 = var_55;
    ret_1 = var_53;
    return;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1107
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frame_linvel_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    bool var_6;
    wp::vec_t<3, wp::float32>* var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 5;
    bool var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 6;
    bool var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 7;
    bool var_21;
    wp::vec_t<3, wp::float32>* var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    const wp::int32 var_32 = 1;
    bool var_33;
    wp::vec_t<3, wp::float32>* var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::mat_t<3, 3, wp::float32>* var_37;
    wp::mat_t<3, 3, wp::float32> var_38;
    wp::mat_t<3, 3, wp::float32> var_39;
    const wp::int32 var_40 = 2;
    bool var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32>* var_45;
    wp::mat_t<3, 3, wp::float32> var_46;
    wp::mat_t<3, 3, wp::float32> var_47;
    const wp::int32 var_48 = 5;
    bool var_49;
    wp::vec_t<3, wp::float32>* var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::mat_t<3, 3, wp::float32>* var_53;
    wp::mat_t<3, 3, wp::float32> var_54;
    wp::mat_t<3, 3, wp::float32> var_55;
    const wp::int32 var_56 = 6;
    bool var_57;
    wp::vec_t<3, wp::float32>* var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::mat_t<3, 3, wp::float32>* var_61;
    wp::mat_t<3, 3, wp::float32> var_62;
    wp::mat_t<3, 3, wp::float32> var_63;
    const wp::int32 var_64 = 7;
    bool var_65;
    wp::vec_t<3, wp::float32>* var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::mat_t<3, 3, wp::float32>* var_69;
    wp::mat_t<3, 3, wp::float32> var_70;
    wp::mat_t<3, 3, wp::float32> var_71;
    const wp::float32 var_72 = 0.0;
    wp::vec_t<3, wp::float32> var_73;
    const wp::int32 var_74 = 3;
    wp::mat_t<3, 3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::mat_t<3, 3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::mat_t<3, 3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::mat_t<3, 3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::mat_t<3, 3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::mat_t<3, 3, wp::float32> var_85;
    wp::vec_t<6, wp::float32> var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<6, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    const wp::int32 var_95 = -1;
    bool var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::mat_t<3, 3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    //---------
    // forward
    // def _frame_linvel(                                                                     <L 1108>
    // if objtype == ObjType.BODY:                                                            <L 1134>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // xpos = xipos_in[worldid, objid]                                                    <L 1135>
        var_2 = wp::address(var_xipos_in, var_worldid, var_objid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 1136>
        var_6 = (var_objtype == var_5);
        if (var_6) {
            // xpos = xpos_in[worldid, objid]                                                 <L 1137>
            var_7 = wp::address(var_xpos_in, var_worldid, var_objid);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
        }
        if (!var_6) {
            // elif objtype == ObjType.GEOM:                                                  <L 1138>
            var_11 = (var_objtype == var_10);
            if (var_11) {
                // xpos = geom_xpos_in[worldid, objid]                                        <L 1139>
                var_12 = wp::address(var_geom_xpos_in, var_worldid, var_objid);
                var_14 = wp::load(var_12);
                var_13 = wp::copy(var_14);
            }
            if (!var_11) {
                // elif objtype == ObjType.SITE:                                              <L 1140>
                var_16 = (var_objtype == var_15);
                if (var_16) {
                    // xpos = site_xpos_in[worldid, objid]                                    <L 1141>
                    var_17 = wp::address(var_site_xpos_in, var_worldid, var_objid);
                    var_19 = wp::load(var_17);
                    var_18 = wp::copy(var_19);
                }
                if (!var_16) {
                    // elif objtype == ObjType.CAMERA:                                        <L 1142>
                    var_21 = (var_objtype == var_20);
                    if (var_21) {
                        // xpos = cam_xpos_in[worldid, objid]                                 <L 1143>
                        var_22 = wp::address(var_cam_xpos_in, var_worldid, var_objid);
                        var_24 = wp::load(var_22);
                        var_23 = wp::copy(var_24);
                    }
                    if (!var_21) {
                        // xpos = wp.vec3(0.0)                                                <L 1145>
                        var_26 = wp::vec_t<3, wp::float32>(var_25);
                    }
                    var_27 = wp::where(var_21, var_23, var_26);
                }
                var_28 = wp::where(var_16, var_18, var_27);
            }
            var_29 = wp::where(var_11, var_13, var_28);
        }
        var_30 = wp::where(var_6, var_8, var_29);
    }
    var_31 = wp::where(var_1, var_3, var_30);
    // if reftype == ObjType.BODY:                                                            <L 1147>
    var_33 = (var_reftype == var_32);
    if (var_33) {
        // xposref = xipos_in[worldid, refid]                                                 <L 1148>
        var_34 = wp::address(var_xipos_in, var_worldid, var_refid);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // xmatref = ximat_in[worldid, refid]                                                 <L 1149>
        var_37 = wp::address(var_ximat_in, var_worldid, var_refid);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
    }
    if (!var_33) {
        // elif reftype == ObjType.XBODY:                                                     <L 1150>
        var_41 = (var_reftype == var_40);
        if (var_41) {
            // xposref = xpos_in[worldid, refid]                                              <L 1151>
            var_42 = wp::address(var_xpos_in, var_worldid, var_refid);
            var_44 = wp::load(var_42);
            var_43 = wp::copy(var_44);
            // xmatref = xmat_in[worldid, refid]                                              <L 1152>
            var_45 = wp::address(var_xmat_in, var_worldid, var_refid);
            var_47 = wp::load(var_45);
            var_46 = wp::copy(var_47);
        }
        if (!var_41) {
            // elif reftype == ObjType.GEOM:                                                  <L 1153>
            var_49 = (var_reftype == var_48);
            if (var_49) {
                // xposref = geom_xpos_in[worldid, refid]                                     <L 1154>
                var_50 = wp::address(var_geom_xpos_in, var_worldid, var_refid);
                var_52 = wp::load(var_50);
                var_51 = wp::copy(var_52);
                // xmatref = geom_xmat_in[worldid, refid]                                     <L 1155>
                var_53 = wp::address(var_geom_xmat_in, var_worldid, var_refid);
                var_55 = wp::load(var_53);
                var_54 = wp::copy(var_55);
            }
            if (!var_49) {
                // elif reftype == ObjType.SITE:                                              <L 1156>
                var_57 = (var_reftype == var_56);
                if (var_57) {
                    // xposref = site_xpos_in[worldid, refid]                                 <L 1157>
                    var_58 = wp::address(var_site_xpos_in, var_worldid, var_refid);
                    var_60 = wp::load(var_58);
                    var_59 = wp::copy(var_60);
                    // xmatref = site_xmat_in[worldid, refid]                                 <L 1158>
                    var_61 = wp::address(var_site_xmat_in, var_worldid, var_refid);
                    var_63 = wp::load(var_61);
                    var_62 = wp::copy(var_63);
                }
                if (!var_57) {
                    // elif reftype == ObjType.CAMERA:                                        <L 1159>
                    var_65 = (var_reftype == var_64);
                    if (var_65) {
                        // xposref = cam_xpos_in[worldid, refid]                              <L 1160>
                        var_66 = wp::address(var_cam_xpos_in, var_worldid, var_refid);
                        var_68 = wp::load(var_66);
                        var_67 = wp::copy(var_68);
                        // xmatref = cam_xmat_in[worldid, refid]                              <L 1161>
                        var_69 = wp::address(var_cam_xmat_in, var_worldid, var_refid);
                        var_71 = wp::load(var_69);
                        var_70 = wp::copy(var_71);
                    }
                    if (!var_65) {
                        // xposref = wp.vec3(0.0)                                             <L 1163>
                        var_73 = wp::vec_t<3, wp::float32>(var_72);
                        // xmatref = wp.identity(3, dtype=float)                              <L 1164>
                        var_75 = wp::identity<3, wp::float32>();
                    }
                    var_76 = wp::where(var_65, var_67, var_73);
                    var_77 = wp::where(var_65, var_70, var_75);
                }
                var_78 = wp::where(var_57, var_59, var_76);
                var_79 = wp::where(var_57, var_62, var_77);
            }
            var_80 = wp::where(var_49, var_51, var_78);
            var_81 = wp::where(var_49, var_54, var_79);
        }
        var_82 = wp::where(var_41, var_43, var_80);
        var_83 = wp::where(var_41, var_46, var_81);
    }
    var_84 = wp::where(var_33, var_35, var_82);
    var_85 = wp::where(var_33, var_38, var_83);
    // cvel, offset = _cvel_offset(                                                           <L 1166>
    // body_rootid,                                                                           <L 1167>
    // geom_bodyid,                                                                           <L 1168>
    // site_bodyid,                                                                           <L 1169>
    // cam_bodyid,                                                                            <L 1170>
    // xpos_in,                                                                               <L 1171>
    // xipos_in,                                                                              <L 1172>
    // geom_xpos_in,                                                                          <L 1173>
    // site_xpos_in,                                                                          <L 1174>
    // cam_xpos_in,                                                                           <L 1175>
    // subtree_com_in,                                                                        <L 1176>
    // cvel_in,                                                                               <L 1177>
    // worldid,                                                                               <L 1178>
    // objtype,                                                                               <L 1179>
    // objid,                                                                                 <L 1180>
    _cvel_offset_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_worldid, var_objtype, var_objid, var_86, var_87);
    // cvelref, offsetref = _cvel_offset(                                                     <L 1182>
    // body_rootid,                                                                           <L 1183>
    // geom_bodyid,                                                                           <L 1184>
    // site_bodyid,                                                                           <L 1185>
    // cam_bodyid,                                                                            <L 1186>
    // xpos_in,                                                                               <L 1187>
    // xipos_in,                                                                              <L 1188>
    // geom_xpos_in,                                                                          <L 1189>
    // site_xpos_in,                                                                          <L 1190>
    // cam_xpos_in,                                                                           <L 1191>
    // subtree_com_in,                                                                        <L 1192>
    // cvel_in,                                                                               <L 1193>
    // worldid,                                                                               <L 1194>
    // reftype,                                                                               <L 1195>
    // refid,                                                                                 <L 1196>
    _cvel_offset_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_worldid, var_reftype, var_refid, var_88, var_89);
    // clinvel = wp.spatial_bottom(cvel)                                                      <L 1198>
    var_90 = wp::spatial_bottom(var_86);
    // cangvel = wp.spatial_top(cvel)                                                         <L 1199>
    var_91 = wp::spatial_top(var_86);
    // cangvelref = wp.spatial_top(cvelref)                                                   <L 1200>
    var_92 = wp::spatial_top(var_88);
    // xlinvel = clinvel - wp.cross(offset, cangvel)                                          <L 1201>
    var_93 = wp::cross(var_87, var_91);
    var_94 = wp::sub(var_90, var_93);
    // if refid > -1:                                                                         <L 1203>
    var_96 = (var_refid > var_95);
    if (var_96) {
        // clinvelref = wp.spatial_bottom(cvelref)                                            <L 1204>
        var_97 = wp::spatial_bottom(var_88);
        // xlinvelref = clinvelref - wp.cross(offsetref, cangvelref)                          <L 1205>
        var_98 = wp::cross(var_89, var_92);
        var_99 = wp::sub(var_97, var_98);
        // rvec = xpos - xposref                                                              <L 1206>
        var_100 = wp::sub(var_31, var_84);
        // rel_vel = xlinvel - xlinvelref + wp.cross(rvec, cangvelref)                        <L 1207>
        var_101 = wp::sub(var_94, var_99);
        var_102 = wp::cross(var_100, var_92);
        var_103 = wp::add(var_101, var_102);
        // return wp.transpose(xmatref) @ rel_vel                                             <L 1208>
        var_104 = wp::transpose(var_85);
        var_105 = wp::mul(var_104, var_103);
        return var_105;
    }
    if (!var_96) {
        // return xlinvel                                                                     <L 1210>
        return var_94;
    }
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1213
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frame_angvel_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype)
{
    //---------
    // primal vars
    wp::vec_t<6, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::int32 var_3 = -1;
    bool var_4;
    const wp::int32 var_5 = 1;
    bool var_6;
    wp::mat_t<3, 3, wp::float32>* var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    const wp::int32 var_10 = 2;
    bool var_11;
    wp::mat_t<3, 3, wp::float32>* var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    const wp::int32 var_15 = 5;
    bool var_16;
    wp::mat_t<3, 3, wp::float32>* var_17;
    wp::mat_t<3, 3, wp::float32> var_18;
    wp::mat_t<3, 3, wp::float32> var_19;
    const wp::int32 var_20 = 6;
    bool var_21;
    wp::mat_t<3, 3, wp::float32>* var_22;
    wp::mat_t<3, 3, wp::float32> var_23;
    wp::mat_t<3, 3, wp::float32> var_24;
    const wp::int32 var_25 = 7;
    bool var_26;
    wp::mat_t<3, 3, wp::float32>* var_27;
    wp::mat_t<3, 3, wp::float32> var_28;
    wp::mat_t<3, 3, wp::float32> var_29;
    const wp::int32 var_30 = 3;
    wp::mat_t<3, 3, wp::float32> var_31;
    wp::mat_t<3, 3, wp::float32> var_32;
    wp::mat_t<3, 3, wp::float32> var_33;
    wp::mat_t<3, 3, wp::float32> var_34;
    wp::mat_t<3, 3, wp::float32> var_35;
    wp::mat_t<3, 3, wp::float32> var_36;
    wp::vec_t<6, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::mat_t<3, 3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    //---------
    // forward
    // def _frame_angvel(                                                                     <L 1214>
    // cvel, _ = _cvel_offset(                                                                <L 1240>
    // body_rootid,                                                                           <L 1241>
    // geom_bodyid,                                                                           <L 1242>
    // site_bodyid,                                                                           <L 1243>
    // cam_bodyid,                                                                            <L 1244>
    // xpos_in,                                                                               <L 1245>
    // xipos_in,                                                                              <L 1246>
    // geom_xpos_in,                                                                          <L 1247>
    // site_xpos_in,                                                                          <L 1248>
    // cam_xpos_in,                                                                           <L 1249>
    // subtree_com_in,                                                                        <L 1250>
    // cvel_in,                                                                               <L 1251>
    // worldid,                                                                               <L 1252>
    // objtype,                                                                               <L 1253>
    // objid,                                                                                 <L 1254>
    _cvel_offset_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_worldid, var_objtype, var_objid, var_0, var_1);
    // cangvel = wp.spatial_top(cvel)                                                         <L 1256>
    var_2 = wp::spatial_top(var_0);
    // if refid > -1:                                                                         <L 1258>
    var_4 = (var_refid > var_3);
    if (var_4) {
        // if reftype == ObjType.BODY:                                                        <L 1259>
        var_6 = (var_reftype == var_5);
        if (var_6) {
            // xmatref = ximat_in[worldid, refid]                                             <L 1260>
            var_7 = wp::address(var_ximat_in, var_worldid, var_refid);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
        }
        if (!var_6) {
            // elif reftype == ObjType.XBODY:                                                 <L 1261>
            var_11 = (var_reftype == var_10);
            if (var_11) {
                // xmatref = xmat_in[worldid, refid]                                          <L 1262>
                var_12 = wp::address(var_xmat_in, var_worldid, var_refid);
                var_14 = wp::load(var_12);
                var_13 = wp::copy(var_14);
            }
            if (!var_11) {
                // elif reftype == ObjType.GEOM:                                              <L 1263>
                var_16 = (var_reftype == var_15);
                if (var_16) {
                    // xmatref = geom_xmat_in[worldid, refid]                                 <L 1264>
                    var_17 = wp::address(var_geom_xmat_in, var_worldid, var_refid);
                    var_19 = wp::load(var_17);
                    var_18 = wp::copy(var_19);
                }
                if (!var_16) {
                    // elif reftype == ObjType.SITE:                                          <L 1265>
                    var_21 = (var_reftype == var_20);
                    if (var_21) {
                        // xmatref = site_xmat_in[worldid, refid]                             <L 1266>
                        var_22 = wp::address(var_site_xmat_in, var_worldid, var_refid);
                        var_24 = wp::load(var_22);
                        var_23 = wp::copy(var_24);
                    }
                    if (!var_21) {
                        // elif reftype == ObjType.CAMERA:                                    <L 1267>
                        var_26 = (var_reftype == var_25);
                        if (var_26) {
                            // xmatref = cam_xmat_in[worldid, refid]                          <L 1268>
                            var_27 = wp::address(var_cam_xmat_in, var_worldid, var_refid);
                            var_29 = wp::load(var_27);
                            var_28 = wp::copy(var_29);
                        }
                        if (!var_26) {
                            // xmatref = wp.identity(3, dtype=float)                          <L 1270>
                            var_31 = wp::identity<3, wp::float32>();
                        }
                        var_32 = wp::where(var_26, var_28, var_31);
                    }
                    var_33 = wp::where(var_21, var_23, var_32);
                }
                var_34 = wp::where(var_16, var_18, var_33);
            }
            var_35 = wp::where(var_11, var_13, var_34);
        }
        var_36 = wp::where(var_6, var_8, var_35);
        // cvelref, _ = _cvel_offset(                                                         <L 1272>
        // body_rootid,                                                                       <L 1273>
        // geom_bodyid,                                                                       <L 1274>
        // site_bodyid,                                                                       <L 1275>
        // cam_bodyid,                                                                        <L 1276>
        // xpos_in,                                                                           <L 1277>
        // xipos_in,                                                                          <L 1278>
        // geom_xpos_in,                                                                      <L 1279>
        // site_xpos_in,                                                                      <L 1280>
        // cam_xpos_in,                                                                       <L 1281>
        // subtree_com_in,                                                                    <L 1282>
        // cvel_in,                                                                           <L 1283>
        // worldid,                                                                           <L 1284>
        // reftype,                                                                           <L 1285>
        // refid,                                                                             <L 1286>
        _cvel_offset_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_worldid, var_reftype, var_refid, var_37, var_38);
        // cangvelref = wp.spatial_top(cvelref)                                               <L 1288>
        var_39 = wp::spatial_top(var_37);
        // return wp.transpose(xmatref) @ (cangvel - cangvelref)                              <L 1290>
        var_40 = wp::transpose(var_36);
        var_41 = wp::sub(var_2, var_39);
        var_42 = wp::mul(var_40, var_41);
        return var_42;
    }
    if (!var_4) {
        // return cangvel                                                                     <L 1292>
        return var_2;
    }
    var_43 = wp::where(var_4, var_38, var_1);
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1295
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _subtree_linvel_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    //---------
    // forward
    // def _subtree_linvel(subtree_linvel_in: wp.array2d[wp.vec3], worldid: int, objid: int) -> wp.vec3:       <L 1296>
    // return subtree_linvel_in[worldid, objid]                                               <L 1297>
    var_0 = wp::address(var_subtree_linvel_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1300
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _subtree_angmom_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    //---------
    // forward
    // def _subtree_angmom(subtree_angmom_in: wp.array2d[wp.vec3], worldid: int, objid: int) -> wp.vec3:       <L 1301>
    // return subtree_angmom_in[worldid, objid]                                               <L 1302>
    var_0 = wp::address(var_subtree_angmom_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:116
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _magnetometer_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_magnetic,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    wp::int32 var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32>* var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _magnetometer(                                                                     <L 117>
    // magnetic = opt_magnetic[worldid % opt_magnetic.shape[0]]                               <L 126>
    var_0 = &(var_opt_magnetic.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    var_4 = wp::mod(var_worldid, var_2);
    var_5 = wp::address(var_opt_magnetic, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // return wp.transpose(site_xmat_in[worldid, objid]) @ magnetic                           <L 127>
    var_8 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_10 = wp::load(var_8);
    var_9 = wp::transpose(var_10);
    var_11 = wp::mul(var_9, var_6);
    return var_11;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:130
static CUDA_CALLABLE wp::vec_t<2, wp::float32> _cam_projection_0(
    wp::array_t<wp::float32> var_cam_fovy,
    wp::array_t<wp::vec_t<2, wp::int32>> var_cam_resolution,
    wp::array_t<wp::vec_t<2, wp::float32>> var_cam_sensorsize,
    wp::array_t<wp::vec_t<4, wp::float32>> var_cam_intrinsic,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_refid)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::float32>* var_0;
    wp::vec_t<2, wp::float32> var_1;
    wp::vec_t<2, wp::float32> var_2;
    wp::shape_t* var_3;
    const wp::int32 var_4 = 0;
    wp::int32 var_5;
    wp::shape_t var_6;
    wp::int32 var_7;
    wp::vec_t<4, wp::float32>* var_8;
    wp::vec_t<4, wp::float32> var_9;
    wp::vec_t<4, wp::float32> var_10;
    wp::shape_t* var_11;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    wp::shape_t var_14;
    wp::int32 var_15;
    wp::float32* var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<2, wp::int32>* var_19;
    wp::vec_t<2, wp::int32> var_20;
    wp::vec_t<2, wp::int32> var_21;
    wp::vec_t<3, wp::float32>* var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::mat_t<3, 3, wp::float32>* var_28;
    wp::mat_t<3, 3, wp::float32> var_29;
    wp::mat_t<3, 3, wp::float32> var_30;
    wp::mat_t<3, 3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    bool var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::float32 var_37 = 0.0;
    bool var_38;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    const wp::float32 var_41 = 0.0;
    bool var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    const wp::float32 var_47 = 1e-15;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::int32 var_50 = 0;
    wp::int32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::int32 var_54 = 1;
    wp::float32 var_55;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::int32 var_60 = 1;
    wp::int32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    const wp::float32 var_64 = 0.5;
    const wp::float32 var_65 = 0.008726646259971648;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 1;
    wp::int32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::int32 var_78 = 0;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 1;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::int32 var_84 = 2;
    wp::float32 var_85;
    wp::float32 var_86;
    bool var_87;
    const wp::float32 var_88 = -1e-15;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::vec_t<2, wp::float32> var_91;
    wp::vec_t<2, wp::float32> var_92;
    const wp::float32 var_93 = 0.5;
    const wp::int32 var_94 = 0;
    wp::int32 var_95;
    wp::float32 var_96;
    const wp::int32 var_97 = 1;
    wp::int32 var_98;
    wp::float32 var_99;
    wp::vec_t<2, wp::float32> var_100;
    wp::vec_t<2, wp::float32> var_101;
    wp::vec_t<2, wp::float32> var_102;
    //---------
    // forward
    // def _cam_projection(                                                                   <L 131>
    // sensorsize = cam_sensorsize[refid]                                                     <L 146>
    var_0 = wp::address(var_cam_sensorsize, var_refid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // intrinsic = cam_intrinsic[worldid % cam_intrinsic.shape[0], refid]                     <L 147>
    var_3 = &(var_cam_intrinsic.shape);
    var_6 = wp::load(var_3);
    var_5 = wp::extract(var_6, var_4);
    var_7 = wp::mod(var_worldid, var_5);
    var_8 = wp::address(var_cam_intrinsic, var_7, var_refid);
    var_10 = wp::load(var_8);
    var_9 = wp::copy(var_10);
    // fovy = cam_fovy[worldid % cam_fovy.shape[0], refid]                                    <L 148>
    var_11 = &(var_cam_fovy.shape);
    var_14 = wp::load(var_11);
    var_13 = wp::extract(var_14, var_12);
    var_15 = wp::mod(var_worldid, var_13);
    var_16 = wp::address(var_cam_fovy, var_15, var_refid);
    var_18 = wp::load(var_16);
    var_17 = wp::copy(var_18);
    // res = cam_resolution[refid]                                                            <L 149>
    var_19 = wp::address(var_cam_resolution, var_refid);
    var_21 = wp::load(var_19);
    var_20 = wp::copy(var_21);
    // target_xpos = site_xpos_in[worldid, objid]                                             <L 151>
    var_22 = wp::address(var_site_xpos_in, var_worldid, var_objid);
    var_24 = wp::load(var_22);
    var_23 = wp::copy(var_24);
    // xpos = cam_xpos_in[worldid, refid]                                                     <L 152>
    var_25 = wp::address(var_cam_xpos_in, var_worldid, var_refid);
    var_27 = wp::load(var_25);
    var_26 = wp::copy(var_27);
    // xmat = cam_xmat_in[worldid, refid]                                                     <L 153>
    var_28 = wp::address(var_cam_xmat_in, var_worldid, var_refid);
    var_30 = wp::load(var_28);
    var_29 = wp::copy(var_30);
    // v = wp.transpose(xmat) @ (target_xpos - xpos)                                          <L 156>
    var_31 = wp::transpose(var_29);
    var_32 = wp::sub(var_23, var_26);
    var_33 = wp::mul(var_31, var_32);
    // if sensorsize[0] != 0.0 and sensorsize[1] != 0.0:                                      <L 159>
    var_36 = wp::extract(var_1, var_35);
    var_38 = (var_36 != var_37);
    var_34 = var_38;
    if (var_34) {
        var_40 = wp::extract(var_1, var_39);
        var_42 = (var_40 != var_41);
        var_34 = var_34 && var_42;
    }
    if (var_34) {
        // fx = intrinsic[0] / (sensorsize[0] + MJ_MINVAL) * float(res[0])                    <L 160>
        var_44 = wp::extract(var_9, var_43);
        var_46 = wp::extract(var_1, var_45);
        var_48 = wp::add(var_46, var_47);
        var_49 = wp::div(var_44, var_48);
        var_51 = wp::extract(var_20, var_50);
        var_52 = wp::float(var_51);
        var_53 = wp::mul(var_49, var_52);
        // fy = intrinsic[1] / (sensorsize[1] + MJ_MINVAL) * float(res[1])                    <L 161>
        var_55 = wp::extract(var_9, var_54);
        var_57 = wp::extract(var_1, var_56);
        var_58 = wp::add(var_57, var_47);
        var_59 = wp::div(var_55, var_58);
        var_61 = wp::extract(var_20, var_60);
        var_62 = wp::float(var_61);
        var_63 = wp::mul(var_59, var_62);
    }
    if (!var_34) {
        // f = 0.5 / wp.tan(fovy * wp.static(wp.pi / 360.0)) * float(res[1])                  <L 163>
        var_66 = wp::mul(var_17, var_65);
        var_67 = wp::tan(var_66);
        var_68 = wp::div(var_64, var_67);
        var_70 = wp::extract(var_20, var_69);
        var_71 = wp::float(var_70);
        var_72 = wp::mul(var_68, var_71);
        // fx = f                                                                             <L 164>
        var_73 = wp::copy(var_72);
        // fy = f                                                                             <L 165>
        var_74 = wp::copy(var_72);
    }
    var_75 = wp::where(var_34, var_53, var_73);
    var_76 = wp::where(var_34, var_63, var_74);
    // pixel_x = -fx * v[0]                                                                   <L 168>
    var_77 = wp::neg(var_75);
    var_79 = wp::extract(var_33, var_78);
    var_80 = wp::mul(var_77, var_79);
    // pixel_y = fy * v[1]                                                                    <L 169>
    var_82 = wp::extract(var_33, var_81);
    var_83 = wp::mul(var_76, var_82);
    // denom = v[2]                                                                           <L 171>
    var_85 = wp::extract(var_33, var_84);
    // if wp.abs(denom) < MJ_MINVAL:                                                          <L 172>
    var_86 = wp::abs(var_85);
    var_87 = (var_86 < var_47);
    if (var_87) {
        // denom = wp.clamp(denom, -MJ_MINVAL, MJ_MINVAL)                                     <L 173>
        var_89 = wp::clamp(var_85, var_88, var_47);
    }
    var_90 = wp::where(var_87, var_89, var_85);
    // return wp.vec2(pixel_x, pixel_y) / denom + 0.5 * wp.vec2(float(res[0]), float(res[1]))       <L 175>
    var_91 = wp::vec_t<2, wp::float32>(var_80, var_83);
    var_92 = wp::div(var_91, var_90);
    var_95 = wp::extract(var_20, var_94);
    var_96 = wp::float(var_95);
    var_98 = wp::extract(var_20, var_97);
    var_99 = wp::float(var_98);
    var_100 = wp::vec_t<2, wp::float32>(var_96, var_99);
    var_101 = wp::mul(var_93, var_100);
    var_102 = wp::add(var_92, var_101);
    return var_102;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<2, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::int32* var_9;
    const wp::int32 var_10 = 41;
    const wp::int32 var_11 = 41;
    wp::int32 var_12;
    bool var_13;
    wp::int32 var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    bool var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::range_t var_35;
    wp::int32 var_36;
    wp::float32 var_37;
    wp::int32 var_38;
    //---------
    // forward
    // def _write_vector(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 14>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 15>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 17>
    var_8 = (var_4 > var_7);
    var_6 = var_8;
    if (var_6) {
        var_9 = wp::address(var_sensor_type, var_sensorid);
        var_12 = wp::int(var_11);
        var_14 = wp::load(var_9);
        var_13 = (var_14 == var_12);
        var_15 = wp::unot(var_13);
        var_6 = var_6 && var_15;
    }
    if (var_6) {
        // datatype = sensor_datatype[sensorid]                                               <L 18>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 19>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // for i in range(sensordim):                                                     <L 20>
            var_21 = wp::range(var_sensordim);
            start_for_0:;
                if (iter_cmp(var_21) == 0) goto end_for_0;
                var_22 = wp::iter_next(var_21);
                // out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)                        <L 21>
                var_23 = wp::extract(var_sensor, var_22);
                var_24 = wp::neg(var_4);
                var_25 = wp::clamp(var_23, var_24, var_4);
                var_26 = wp::add(var_1, var_22);
                wp::array_store(var_out, var_26, var_25);
                goto start_for_0;
            end_for_0:;
            // return                                                                         <L 22>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 23>
            var_28 = (var_17 == var_27);
            if (var_28) {
                // for i in range(sensordim):                                                 <L 24>
                var_29 = wp::range(var_sensordim);
                start_for_3:;
                    if (iter_cmp(var_29) == 0) goto end_for_3;
                    var_30 = wp::iter_next(var_29);
                    // out[adr + i] = wp.min(sensor[i], cutoff)                               <L 25>
                    var_31 = wp::extract(var_sensor, var_30);
                    var_32 = wp::min(var_31, var_4);
                    var_33 = wp::add(var_1, var_30);
                    wp::array_store(var_out, var_33, var_32);
                    goto start_for_3;
                end_for_3:;
                // return                                                                     <L 26>
                return;
            }
        }
        var_34 = wp::where(var_20, var_22, var_30);
    }
    // for i in range(sensordim):                                                             <L 28>
    var_35 = wp::range(var_sensordim);
    start_for_6:;
        if (iter_cmp(var_35) == 0) goto end_for_6;
        var_36 = wp::iter_next(var_35);
        // out[adr + i] = sensor[i]                                                           <L 29>
        var_37 = wp::extract(var_sensor, var_36);
        var_38 = wp::add(var_1, var_36);
        wp::array_store(var_out, var_38, var_37);
        goto start_for_6;
    end_for_6:;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:200
static CUDA_CALLABLE wp::float32 _joint_pos_0(
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::float32* var_1;
    wp::int32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def _joint_pos(jnt_qposadr: wp.array[int], qpos_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 201>
    // return qpos_in[worldid, jnt_qposadr[objid]]                                            <L 202>
    var_0 = wp::address(var_jnt_qposadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::address(var_qpos_in, var_worldid, var_2);
    var_4 = wp::load(var_1);
    var_3 = wp::copy(var_4);
    return var_3;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:205
static CUDA_CALLABLE wp::float32 _tendon_pos_0(
    wp::array_t<wp::float32> var_ten_length_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _tendon_pos(ten_length_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 206>
    // return ten_length_in[worldid, objid]                                                   <L 207>
    var_0 = wp::address(var_ten_length_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:210
static CUDA_CALLABLE wp::float32 _actuator_pos_0(
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _actuator_pos(actuator_length_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 211>
    // return actuator_length_in[worldid, objid]                                              <L 212>
    var_0 = wp::address(var_actuator_length_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:215
static CUDA_CALLABLE wp::quat_t<wp::float32> _ball_quat_0(
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::float32* var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::float32* var_8;
    const wp::int32 var_9 = 2;
    wp::int32 var_10;
    wp::float32* var_11;
    const wp::int32 var_12 = 3;
    wp::int32 var_13;
    wp::float32* var_14;
    wp::quat_t<wp::float32> var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::quat_t<wp::float32> var_20;
    //---------
    // forward
    // def _ball_quat(jnt_qposadr: wp.array[int], qpos_in: wp.array2d[float], worldid: int, objid: int) -> wp.quat:       <L 216>
    // adr = jnt_qposadr[objid]                                                               <L 217>
    var_0 = wp::address(var_jnt_qposadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // quat = wp.quat(                                                                        <L 218>
    // qpos_in[worldid, adr + 0],                                                             <L 219>
    var_4 = wp::add(var_1, var_3);
    var_5 = wp::address(var_qpos_in, var_worldid, var_4);
    // qpos_in[worldid, adr + 1],                                                             <L 220>
    var_7 = wp::add(var_1, var_6);
    var_8 = wp::address(var_qpos_in, var_worldid, var_7);
    // qpos_in[worldid, adr + 2],                                                             <L 221>
    var_10 = wp::add(var_1, var_9);
    var_11 = wp::address(var_qpos_in, var_worldid, var_10);
    // qpos_in[worldid, adr + 3],                                                             <L 222>
    var_13 = wp::add(var_1, var_12);
    var_14 = wp::address(var_qpos_in, var_worldid, var_13);
    var_16 = wp::load(var_5);
    var_17 = wp::load(var_8);
    var_18 = wp::load(var_11);
    var_19 = wp::load(var_14);
    var_15 = wp::quat_t<wp::float32>(var_16, var_17, var_18, var_19);
    // return wp.normalize(quat)                                                              <L 224>
    var_20 = wp::normalize(var_15);
    return var_20;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::quat_t<wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::int32* var_9;
    const wp::int32 var_10 = 41;
    const wp::int32 var_11 = 41;
    wp::int32 var_12;
    bool var_13;
    wp::int32 var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    bool var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::range_t var_35;
    wp::int32 var_36;
    wp::float32 var_37;
    wp::int32 var_38;
    //---------
    // forward
    // def _write_vector(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 14>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 15>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 17>
    var_8 = (var_4 > var_7);
    var_6 = var_8;
    if (var_6) {
        var_9 = wp::address(var_sensor_type, var_sensorid);
        var_12 = wp::int(var_11);
        var_14 = wp::load(var_9);
        var_13 = (var_14 == var_12);
        var_15 = wp::unot(var_13);
        var_6 = var_6 && var_15;
    }
    if (var_6) {
        // datatype = sensor_datatype[sensorid]                                               <L 18>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 19>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // for i in range(sensordim):                                                     <L 20>
            var_21 = wp::range(var_sensordim);
            start_for_0:;
                if (iter_cmp(var_21) == 0) goto end_for_0;
                var_22 = wp::iter_next(var_21);
                // out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)                        <L 21>
                var_23 = wp::extract(var_sensor, var_22);
                var_24 = wp::neg(var_4);
                var_25 = wp::clamp(var_23, var_24, var_4);
                var_26 = wp::add(var_1, var_22);
                wp::array_store(var_out, var_26, var_25);
                goto start_for_0;
            end_for_0:;
            // return                                                                         <L 22>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 23>
            var_28 = (var_17 == var_27);
            if (var_28) {
                // for i in range(sensordim):                                                 <L 24>
                var_29 = wp::range(var_sensordim);
                start_for_3:;
                    if (iter_cmp(var_29) == 0) goto end_for_3;
                    var_30 = wp::iter_next(var_29);
                    // out[adr + i] = wp.min(sensor[i], cutoff)                               <L 25>
                    var_31 = wp::extract(var_sensor, var_30);
                    var_32 = wp::min(var_31, var_4);
                    var_33 = wp::add(var_1, var_30);
                    wp::array_store(var_out, var_33, var_32);
                    goto start_for_3;
                end_for_3:;
                // return                                                                     <L 26>
                return;
            }
        }
        var_34 = wp::where(var_20, var_22, var_30);
    }
    // for i in range(sensordim):                                                             <L 28>
    var_35 = wp::range(var_sensordim);
    start_for_6:;
        if (iter_cmp(var_35) == 0) goto end_for_6;
        var_36 = wp::iter_next(var_35);
        // out[adr + i] = sensor[i]                                                           <L 29>
        var_37 = wp::extract(var_sensor, var_36);
        var_38 = wp::add(var_1, var_36);
        wp::array_store(var_out, var_38, var_37);
        goto start_for_6;
    end_for_6:;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:265
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _get_pos_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    bool var_6;
    wp::vec_t<3, wp::float32>* var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 5;
    bool var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 6;
    bool var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 7;
    bool var_21;
    wp::vec_t<3, wp::float32>* var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    wp::vec_t<3, wp::float32> var_26;
    //---------
    // forward
    // def _get_pos(                                                                          <L 266>
    // if objtype == ObjType.BODY:                                                            <L 278>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // return xipos_in[worldid, objid]                                                    <L 279>
        var_2 = wp::address(var_xipos_in, var_worldid, var_objid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        return var_3;
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 280>
        var_6 = (var_objtype == var_5);
        if (var_6) {
            // return xpos_in[worldid, objid]                                                 <L 281>
            var_7 = wp::address(var_xpos_in, var_worldid, var_objid);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            return var_8;
        }
        if (!var_6) {
            // elif objtype == ObjType.GEOM:                                                  <L 282>
            var_11 = (var_objtype == var_10);
            if (var_11) {
                // return geom_xpos_in[worldid, objid]                                        <L 283>
                var_12 = wp::address(var_geom_xpos_in, var_worldid, var_objid);
                var_14 = wp::load(var_12);
                var_13 = wp::copy(var_14);
                return var_13;
            }
            if (!var_11) {
                // elif objtype == ObjType.SITE:                                              <L 284>
                var_16 = (var_objtype == var_15);
                if (var_16) {
                    // return site_xpos_in[worldid, objid]                                    <L 285>
                    var_17 = wp::address(var_site_xpos_in, var_worldid, var_objid);
                    var_19 = wp::load(var_17);
                    var_18 = wp::copy(var_19);
                    return var_18;
                }
                if (!var_16) {
                    // elif objtype == ObjType.CAMERA:                                        <L 286>
                    var_21 = (var_objtype == var_20);
                    if (var_21) {
                        // return cam_xpos_in[worldid, objid]                                 <L 287>
                        var_22 = wp::address(var_cam_xpos_in, var_worldid, var_objid);
                        var_24 = wp::load(var_22);
                        var_23 = wp::copy(var_24);
                        return var_23;
                    }
                    if (!var_21) {
                        // return wp.vec3(0.0)                                                <L 289>
                        var_26 = wp::vec_t<3, wp::float32>(var_25);
                        return var_26;
                    }
                }
            }
        }
    }
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:292
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> _get_mat_0(
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::mat_t<3, 3, wp::float32>* var_2;
    wp::mat_t<3, 3, wp::float32> var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    bool var_6;
    wp::mat_t<3, 3, wp::float32>* var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    const wp::int32 var_10 = 5;
    bool var_11;
    wp::mat_t<3, 3, wp::float32>* var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    const wp::int32 var_15 = 6;
    bool var_16;
    wp::mat_t<3, 3, wp::float32>* var_17;
    wp::mat_t<3, 3, wp::float32> var_18;
    wp::mat_t<3, 3, wp::float32> var_19;
    const wp::int32 var_20 = 7;
    bool var_21;
    wp::mat_t<3, 3, wp::float32>* var_22;
    wp::mat_t<3, 3, wp::float32> var_23;
    wp::mat_t<3, 3, wp::float32> var_24;
    const wp::int32 var_25 = 3;
    wp::mat_t<3, 3, wp::float32> var_26;
    //---------
    // forward
    // def _get_mat(                                                                          <L 293>
    // if objtype == ObjType.BODY:                                                            <L 305>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // return ximat_in[worldid, objid]                                                    <L 306>
        var_2 = wp::address(var_ximat_in, var_worldid, var_objid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        return var_3;
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 307>
        var_6 = (var_objtype == var_5);
        if (var_6) {
            // return xmat_in[worldid, objid]                                                 <L 308>
            var_7 = wp::address(var_xmat_in, var_worldid, var_objid);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            return var_8;
        }
        if (!var_6) {
            // elif objtype == ObjType.GEOM:                                                  <L 309>
            var_11 = (var_objtype == var_10);
            if (var_11) {
                // return geom_xmat_in[worldid, objid]                                        <L 310>
                var_12 = wp::address(var_geom_xmat_in, var_worldid, var_objid);
                var_14 = wp::load(var_12);
                var_13 = wp::copy(var_14);
                return var_13;
            }
            if (!var_11) {
                // elif objtype == ObjType.SITE:                                              <L 311>
                var_16 = (var_objtype == var_15);
                if (var_16) {
                    // return site_xmat_in[worldid, objid]                                    <L 312>
                    var_17 = wp::address(var_site_xmat_in, var_worldid, var_objid);
                    var_19 = wp::load(var_17);
                    var_18 = wp::copy(var_19);
                    return var_18;
                }
                if (!var_16) {
                    // elif objtype == ObjType.CAMERA:                                        <L 313>
                    var_21 = (var_objtype == var_20);
                    if (var_21) {
                        // return cam_xmat_in[worldid, objid]                                 <L 314>
                        var_22 = wp::address(var_cam_xmat_in, var_worldid, var_objid);
                        var_24 = wp::load(var_22);
                        var_23 = wp::copy(var_24);
                        return var_23;
                    }
                    if (!var_21) {
                        // return wp.identity(3, dtype=wp.float32)                            <L 316>
                        var_26 = wp::identity<3, wp::float32>();
                        return var_26;
                    }
                }
            }
        }
    }
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:376
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frame_pos_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = -1;
    bool var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _frame_pos(                                                                        <L 377>
    // xpos = _get_pos(xpos_in, xipos_in, geom_xpos_in, site_xpos_in, cam_xpos_in, worldid, objtype, objid)       <L 396>
    var_0 = _get_pos_0(var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_worldid, var_objtype, var_objid);
    // if refid == -1:                                                                        <L 397>
    var_2 = (var_refid == var_1);
    if (var_2) {
        // return xpos                                                                        <L 398>
        return var_0;
    }
    // xpos_ref = _get_pos(xpos_in, xipos_in, geom_xpos_in, site_xpos_in, cam_xpos_in, worldid, reftype, refid)       <L 400>
    var_3 = _get_pos_0(var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_worldid, var_reftype, var_refid);
    // xmat_ref = _get_mat(xmat_in, ximat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, reftype, refid)       <L 401>
    var_4 = _get_mat_0(var_xmat_in, var_ximat_in, var_geom_xmat_in, var_site_xmat_in, var_cam_xmat_in, var_worldid, var_reftype, var_refid);
    // return wp.transpose(xmat_ref) @ (xpos - xpos_ref)                                      <L 402>
    var_5 = wp::transpose(var_4);
    var_6 = wp::sub(var_0, var_3);
    var_7 = wp::mul(var_5, var_6);
    return var_7;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:405
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frame_axis_0(
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::int32 var_frame_axis)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = -1;
    bool var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::mat_t<3, 3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    //---------
    // forward
    // def _frame_axis(                                                                       <L 406>
    // xmat = _get_mat(xmat_in, ximat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, objtype, objid)       <L 421>
    var_0 = _get_mat_0(var_xmat_in, var_ximat_in, var_geom_xmat_in, var_site_xmat_in, var_cam_xmat_in, var_worldid, var_objtype, var_objid);
    // axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])          <L 422>
    var_2 = wp::extract(var_0, var_1, var_frame_axis);
    var_4 = wp::extract(var_0, var_3, var_frame_axis);
    var_6 = wp::extract(var_0, var_5, var_frame_axis);
    var_7 = wp::vec_t<3, wp::float32>(var_2, var_4, var_6);
    // if refid == -1:                                                                        <L 424>
    var_9 = (var_refid == var_8);
    if (var_9) {
        // return axis                                                                        <L 425>
        return var_7;
    }
    // xmat_ref = _get_mat(xmat_in, ximat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, reftype, refid)       <L 427>
    var_10 = _get_mat_0(var_xmat_in, var_ximat_in, var_geom_xmat_in, var_site_xmat_in, var_cam_xmat_in, var_worldid, var_reftype, var_refid);
    // return wp.transpose(xmat_ref) @ axis                                                   <L 428>
    var_11 = wp::transpose(var_10);
    var_12 = wp::mul(var_11, var_7);
    return var_12;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:341
static CUDA_CALLABLE wp::quat_t<wp::float32> _get_quat_0(
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::int32 var_6;
    wp::quat_t<wp::float32>* var_7;
    wp::quat_t<wp::float32>* var_8;
    wp::quat_t<wp::float32> var_9;
    wp::quat_t<wp::float32> var_10;
    wp::quat_t<wp::float32> var_11;
    const wp::int32 var_12 = 2;
    bool var_13;
    wp::quat_t<wp::float32>* var_14;
    wp::quat_t<wp::float32> var_15;
    wp::quat_t<wp::float32> var_16;
    const wp::int32 var_17 = 5;
    bool var_18;
    wp::shape_t* var_19;
    const wp::int32 var_20 = 0;
    wp::int32 var_21;
    wp::shape_t var_22;
    wp::int32 var_23;
    wp::int32* var_24;
    wp::quat_t<wp::float32>* var_25;
    wp::int32 var_26;
    wp::quat_t<wp::float32>* var_27;
    wp::quat_t<wp::float32> var_28;
    wp::quat_t<wp::float32> var_29;
    wp::quat_t<wp::float32> var_30;
    const wp::int32 var_31 = 6;
    bool var_32;
    wp::shape_t* var_33;
    const wp::int32 var_34 = 0;
    wp::int32 var_35;
    wp::shape_t var_36;
    wp::int32 var_37;
    wp::int32* var_38;
    wp::quat_t<wp::float32>* var_39;
    wp::int32 var_40;
    wp::quat_t<wp::float32>* var_41;
    wp::quat_t<wp::float32> var_42;
    wp::quat_t<wp::float32> var_43;
    wp::quat_t<wp::float32> var_44;
    const wp::int32 var_45 = 7;
    bool var_46;
    wp::shape_t* var_47;
    const wp::int32 var_48 = 0;
    wp::int32 var_49;
    wp::shape_t var_50;
    wp::int32 var_51;
    wp::int32* var_52;
    wp::quat_t<wp::float32>* var_53;
    wp::int32 var_54;
    wp::quat_t<wp::float32>* var_55;
    wp::quat_t<wp::float32> var_56;
    wp::quat_t<wp::float32> var_57;
    wp::quat_t<wp::float32> var_58;
    const wp::float32 var_59 = 1.0;
    const wp::float32 var_60 = 0.0;
    const wp::float32 var_61 = 0.0;
    const wp::float32 var_62 = 0.0;
    wp::quat_t<wp::float32> var_63;
    //---------
    // forward
    // def _get_quat(                                                                         <L 342>
    // if objtype == ObjType.BODY:                                                            <L 358>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // body_iquat_id = worldid % body_iquat.shape[0]                                      <L 359>
        var_2 = &(var_body_iquat.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_worldid, var_4);
        // return math.mul_quat(xquat_in[worldid, objid], body_iquat[body_iquat_id, objid])       <L 360>
        var_7 = wp::address(var_xquat_in, var_worldid, var_objid);
        var_8 = wp::address(var_body_iquat, var_6, var_objid);
        var_10 = wp::load(var_7);
        var_11 = wp::load(var_8);
        var_9 = mul_quat_0(var_10, var_11);
        return var_9;
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 361>
        var_13 = (var_objtype == var_12);
        if (var_13) {
            // return xquat_in[worldid, objid]                                                <L 362>
            var_14 = wp::address(var_xquat_in, var_worldid, var_objid);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            return var_15;
        }
        if (!var_13) {
            // elif objtype == ObjType.GEOM:                                                  <L 363>
            var_18 = (var_objtype == var_17);
            if (var_18) {
                // geom_quat_id = worldid % geom_quat.shape[0]                                <L 364>
                var_19 = &(var_geom_quat.shape);
                var_22 = wp::load(var_19);
                var_21 = wp::extract(var_22, var_20);
                var_23 = wp::mod(var_worldid, var_21);
                // return math.mul_quat(xquat_in[worldid, geom_bodyid[objid]], geom_quat[geom_quat_id, objid])       <L 365>
                var_24 = wp::address(var_geom_bodyid, var_objid);
                var_26 = wp::load(var_24);
                var_25 = wp::address(var_xquat_in, var_worldid, var_26);
                var_27 = wp::address(var_geom_quat, var_23, var_objid);
                var_29 = wp::load(var_25);
                var_30 = wp::load(var_27);
                var_28 = mul_quat_0(var_29, var_30);
                return var_28;
            }
            if (!var_18) {
                // elif objtype == ObjType.SITE:                                              <L 366>
                var_32 = (var_objtype == var_31);
                if (var_32) {
                    // site_quat_id = worldid % site_quat.shape[0]                            <L 367>
                    var_33 = &(var_site_quat.shape);
                    var_36 = wp::load(var_33);
                    var_35 = wp::extract(var_36, var_34);
                    var_37 = wp::mod(var_worldid, var_35);
                    // return math.mul_quat(xquat_in[worldid, site_bodyid[objid]], site_quat[site_quat_id, objid])       <L 368>
                    var_38 = wp::address(var_site_bodyid, var_objid);
                    var_40 = wp::load(var_38);
                    var_39 = wp::address(var_xquat_in, var_worldid, var_40);
                    var_41 = wp::address(var_site_quat, var_37, var_objid);
                    var_43 = wp::load(var_39);
                    var_44 = wp::load(var_41);
                    var_42 = mul_quat_0(var_43, var_44);
                    return var_42;
                }
                if (!var_32) {
                    // elif objtype == ObjType.CAMERA:                                        <L 369>
                    var_46 = (var_objtype == var_45);
                    if (var_46) {
                        // cam_quat_id = worldid % cam_quat.shape[0]                          <L 370>
                        var_47 = &(var_cam_quat.shape);
                        var_50 = wp::load(var_47);
                        var_49 = wp::extract(var_50, var_48);
                        var_51 = wp::mod(var_worldid, var_49);
                        // return math.mul_quat(xquat_in[worldid, cam_bodyid[objid]], cam_quat[cam_quat_id, objid])       <L 371>
                        var_52 = wp::address(var_cam_bodyid, var_objid);
                        var_54 = wp::load(var_52);
                        var_53 = wp::address(var_xquat_in, var_worldid, var_54);
                        var_55 = wp::address(var_cam_quat, var_51, var_objid);
                        var_57 = wp::load(var_53);
                        var_58 = wp::load(var_55);
                        var_56 = mul_quat_0(var_57, var_58);
                        return var_56;
                    }
                    if (!var_46) {
                        // return wp.quat(1.0, 0.0, 0.0, 0.0)                                 <L 373>
                        var_63 = wp::quat_t<wp::float32>(var_59, var_60, var_61, var_62);
                        return var_63;
                    }
                }
            }
        }
    }
    return {};
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_inv_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 3;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::quat_t<wp::float32> var_11;
    //---------
    // forward
    // def quat_inv(quat: wp.quat) -> wp.quat:                                                <L 116>
    // return wp.quat(quat[0], -quat[1], -quat[2], -quat[3])                                  <L 117>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_quat, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_quat, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    return var_11;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:431
static CUDA_CALLABLE wp::quat_t<wp::float32> _frame_quat_0(
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype)
{
    //---------
    // primal vars
    wp::quat_t<wp::float32> var_0;
    const wp::int32 var_1 = -1;
    bool var_2;
    wp::quat_t<wp::float32> var_3;
    wp::quat_t<wp::float32> var_4;
    wp::quat_t<wp::float32> var_5;
    //---------
    // forward
    // def _frame_quat(                                                                       <L 432>
    // quat = _get_quat(                                                                      <L 450>
    // body_iquat,                                                                            <L 451>
    // geom_bodyid,                                                                           <L 452>
    // geom_quat,                                                                             <L 453>
    // site_bodyid,                                                                           <L 454>
    // site_quat,                                                                             <L 455>
    // cam_bodyid,                                                                            <L 456>
    // cam_quat,                                                                              <L 457>
    // xquat_in,                                                                              <L 458>
    // worldid,                                                                               <L 459>
    // objtype,                                                                               <L 460>
    // objid,                                                                                 <L 461>
    var_0 = _get_quat_0(var_body_iquat, var_geom_bodyid, var_geom_quat, var_site_bodyid, var_site_quat, var_cam_bodyid, var_cam_quat, var_xquat_in, var_worldid, var_objtype, var_objid);
    // if refid == -1:                                                                        <L 464>
    var_2 = (var_refid == var_1);
    if (var_2) {
        // return quat                                                                        <L 465>
        return var_0;
    }
    // refquat = _get_quat(                                                                   <L 467>
    // body_iquat,                                                                            <L 468>
    // geom_bodyid,                                                                           <L 469>
    // geom_quat,                                                                             <L 470>
    // site_bodyid,                                                                           <L 471>
    // site_quat,                                                                             <L 472>
    // cam_bodyid,                                                                            <L 473>
    // cam_quat,                                                                              <L 474>
    // xquat_in,                                                                              <L 475>
    // worldid,                                                                               <L 476>
    // reftype,                                                                               <L 477>
    // refid,                                                                                 <L 478>
    var_3 = _get_quat_0(var_body_iquat, var_geom_bodyid, var_geom_quat, var_site_bodyid, var_site_quat, var_cam_bodyid, var_cam_quat, var_xquat_in, var_worldid, var_reftype, var_refid);
    // return math.mul_quat(math.quat_inv(refquat), quat)                                     <L 481>
    var_4 = quat_inv_0(var_3);
    var_5 = mul_quat_0(var_4, var_0);
    return var_5;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:484
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _subtree_com_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    //---------
    // forward
    // def _subtree_com(subtree_com_in: wp.array2d[wp.vec3], worldid: int, objid: int) -> wp.vec3:       <L 485>
    // return subtree_com_in[worldid, objid]                                                  <L 486>
    var_0 = wp::address(var_subtree_com_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<6, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::int32* var_9;
    const wp::int32 var_10 = 41;
    const wp::int32 var_11 = 41;
    wp::int32 var_12;
    bool var_13;
    wp::int32 var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    bool var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::range_t var_35;
    wp::int32 var_36;
    wp::float32 var_37;
    wp::int32 var_38;
    //---------
    // forward
    // def _write_vector(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 14>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 15>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 17>
    var_8 = (var_4 > var_7);
    var_6 = var_8;
    if (var_6) {
        var_9 = wp::address(var_sensor_type, var_sensorid);
        var_12 = wp::int(var_11);
        var_14 = wp::load(var_9);
        var_13 = (var_14 == var_12);
        var_15 = wp::unot(var_13);
        var_6 = var_6 && var_15;
    }
    if (var_6) {
        // datatype = sensor_datatype[sensorid]                                               <L 18>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 19>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // for i in range(sensordim):                                                     <L 20>
            var_21 = wp::range(var_sensordim);
            start_for_0:;
                if (iter_cmp(var_21) == 0) goto end_for_0;
                var_22 = wp::iter_next(var_21);
                // out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)                        <L 21>
                var_23 = wp::extract(var_sensor, var_22);
                var_24 = wp::neg(var_4);
                var_25 = wp::clamp(var_23, var_24, var_4);
                var_26 = wp::add(var_1, var_22);
                wp::array_store(var_out, var_26, var_25);
                goto start_for_0;
            end_for_0:;
            // return                                                                         <L 22>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 23>
            var_28 = (var_17 == var_27);
            if (var_28) {
                // for i in range(sensordim):                                                 <L 24>
                var_29 = wp::range(var_sensordim);
                start_for_3:;
                    if (iter_cmp(var_29) == 0) goto end_for_3;
                    var_30 = wp::iter_next(var_29);
                    // out[adr + i] = wp.min(sensor[i], cutoff)                               <L 25>
                    var_31 = wp::extract(var_sensor, var_30);
                    var_32 = wp::min(var_31, var_4);
                    var_33 = wp::add(var_1, var_30);
                    wp::array_store(var_out, var_33, var_32);
                    goto start_for_3;
                end_for_3:;
                // return                                                                     <L 26>
                return;
            }
        }
        var_34 = wp::where(var_20, var_22, var_30);
    }
    // for i in range(sensordim):                                                             <L 28>
    var_35 = wp::range(var_sensordim);
    start_for_6:;
        if (iter_cmp(var_35) == 0) goto end_for_6;
        var_36 = wp::iter_next(var_35);
        // out[adr + i] = sensor[i]                                                           <L 29>
        var_37 = wp::extract(var_sensor, var_36);
        var_38 = wp::add(var_1, var_36);
        wp::array_store(var_out, var_38, var_37);
        goto start_for_6;
    end_for_6:;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:489
static CUDA_CALLABLE wp::float32 _clock_0(
    wp::array_t<wp::float32> var_time_in,
    wp::int32 var_worldid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _clock(time_in: wp.array[float], worldid: int) -> float:                           <L 490>
    // return time_in[worldid]                                                                <L 491>
    var_0 = wp::address(var_time_in, var_worldid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_1(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:32
static CUDA_CALLABLE void adj__ray_map_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:213
static CUDA_CALLABLE void adj_ray_plane_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
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
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:105
static CUDA_CALLABLE void adj__ray_quad_0(
    wp::float32 var_a,
    wp::float32 var_b,
    wp::float32 var_c,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::float32 & adj_a,
    wp::float32 & adj_b,
    wp::float32 & adj_c,
    wp::float32 & adj_ret_0,
    wp::vec_t<2, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:237
static CUDA_CALLABLE void adj_ray_sphere_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::float32 var_dist_sqr,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::float32 & adj_dist_sqr,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:254
static CUDA_CALLABLE void adj_ray_capsule_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:328
static CUDA_CALLABLE void adj_ray_ellipsoid_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:359
static CUDA_CALLABLE void adj_ray_cylinder_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:420
static CUDA_CALLABLE void adj_ray_box_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<6, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<6, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:798
static CUDA_CALLABLE void adj_ray_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_geomtype,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::int32 & adj_geomtype,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_scalar_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::float32 var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::float32 & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:726
static CUDA_CALLABLE void adj_poly_potential_0(
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:44
static CUDA_CALLABLE void adj_rot_vec_quat_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::quat_t<wp::float32> var_quat,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:81
static CUDA_CALLABLE void adj_get_sdf_params_0(
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::float32>> var_oct_coeff,
    wp::array_t<wp::int32> var_mesh_octadr,
    wp::array_t<wp::int32> var_plugin,
    wp::array_t<wp::vec_t<128, wp::float32>> var_plugin_attr,
    wp::int32 var_g_type,
    wp::vec_t<3, wp::float32> var_g_size,
    wp::int32 var_plugin_id,
    wp::int32 var_mesh_id,
    wp::vec_t<128, wp::float32> & ret_0,
    wp::int32 & ret_1,
    VolumeData_53ac1a2d & ret_2,
    MeshData_52eaa0fa & ret_3,
    wp::array_t<wp::vec_t<8, wp::int32>> & adj_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::float32>> & adj_oct_coeff,
    wp::array_t<wp::int32> & adj_mesh_octadr,
    wp::array_t<wp::int32> & adj_plugin,
    wp::array_t<wp::vec_t<128, wp::float32>> & adj_plugin_attr,
    wp::int32 & adj_g_type,
    wp::vec_t<3, wp::float32> & adj_g_size,
    wp::int32 & adj_plugin_id,
    wp::int32 & adj_mesh_id,
    wp::vec_t<128, wp::float32> & adj_ret_0,
    wp::int32 & adj_ret_1,
    VolumeData_53ac1a2d & adj_ret_2,
    MeshData_52eaa0fa & adj_ret_3)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:158
static CUDA_CALLABLE void adj_sphere_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:187
static CUDA_CALLABLE void adj_capsule_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:196
static CUDA_CALLABLE void adj_cylinder_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:148
static CUDA_CALLABLE void adj_radial_field_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:163
static CUDA_CALLABLE void adj_box_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:175
static CUDA_CALLABLE void adj_ellipsoid_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:128
static CUDA_CALLABLE void adj__orthogonal_basis_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:154
static CUDA_CALLABLE void adj__ray_triangle_1(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> var_b0,
    wp::vec_t<3, wp::float32> var_b1,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_v0,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::vec_t<3, wp::float32> & adj_b0,
    wp::vec_t<3, wp::float32> & adj_b1,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:628
static CUDA_CALLABLE void adj_ray_mesh_0(
    wp::int32 var_nmeshface,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_faceadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::vec_t<3, wp::int32>> var_mesh_face,
    wp::int32 var_data_id,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::int32 & adj_nmeshface,
    wp::array_t<wp::int32> & adj_mesh_vertadr,
    wp::array_t<wp::int32> & adj_mesh_faceadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_vert,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_mesh_face,
    wp::int32 & adj_data_id,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:391
static CUDA_CALLABLE void adj_box_project_0(
    wp::vec_t<3, wp::float32> var_center,
    wp::vec_t<3, wp::float32> var_half_size,
    wp::vec_t<3, wp::float32> var_xyz,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_center,
    wp::vec_t<3, wp::float32> & adj_half_size,
    wp::vec_t<3, wp::float32> & adj_xyz,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:318
static CUDA_CALLABLE void adj_find_oct_0(
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::vec_t<3, wp::float32> var_p,
    bool var_grad,
    wp::int32 var_root,
    wp::int32 & ret_0,
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> & ret_1,
    wp::array_t<wp::vec_t<8, wp::int32>> & adj_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_oct_aabb,
    wp::vec_t<3, wp::float32> & adj_p,
    bool & adj_grad,
    wp::int32 & adj_root,
    wp::int32 & adj_ret_0,
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:428
static CUDA_CALLABLE void adj_sample_volume_sdf_0(
    wp::vec_t<3, wp::float32> var_xyz,
    VolumeData_53ac1a2d var_volume_data,
    wp::vec_t<3, wp::float32> & adj_xyz,
    VolumeData_53ac1a2d & adj_volume_data,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:298
static CUDA_CALLABLE void adj_user_sdf_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<128, wp::float32> var_attr,
    wp::int32 var_sdf_type,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<128, wp::float32> & adj_attr,
    wp::int32 & adj_sdf_type,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/collision_sdf.py:455
static CUDA_CALLABLE void adj_sdf_0(
    wp::int32 var_type,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<128, wp::float32> var_attr,
    wp::int32 var_sdf_type,
    VolumeData_53ac1a2d var_volume_data,
    MeshData_52eaa0fa var_mesh_data,
    wp::int32 & adj_type,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<128, wp::float32> & adj_attr,
    wp::int32 & adj_sdf_type,
    VolumeData_53ac1a2d & adj_volume_data,
    MeshData_52eaa0fa & adj_mesh_data,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:2138
static CUDA_CALLABLE void adj__transform_spatial_0(
    wp::vec_t<6, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> var_dif,
    wp::vec_t<6, wp::float32> & adj_vec,
    wp::vec_t<3, wp::float32> & adj_dif,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/support.py:325
static CUDA_CALLABLE void adj__decode_pyramid_0(
    wp::int32 var_njmax_in,
    wp::array_t<wp::float32> var_pyramid,
    wp::int32 var_efc_address,
    wp::vec_t<5, wp::float32> var_mu,
    wp::int32 var_condim,
    wp::int32 & adj_njmax_in,
    wp::array_t<wp::float32> & adj_pyramid,
    wp::int32 & adj_efc_address,
    wp::vec_t<5, wp::float32> & adj_mu,
    wp::int32 & adj_condim,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/support.py:351
static CUDA_CALLABLE void adj_contact_force_fn_0(
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::int32 var_worldid,
    wp::int32 var_contact_id,
    bool var_to_world_frame,
    wp::int32 & adj_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_in,
    wp::array_t<wp::int32> & adj_contact_dim_in,
    wp::array_t<wp::int32> & adj_contact_efc_address_in,
    wp::array_t<wp::float32> & adj_efc_force_in,
    wp::int32 & adj_njmax_in,
    wp::array_t<wp::int32> & adj_nacon_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_contact_id,
    bool & adj_to_world_frame,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1509
static CUDA_CALLABLE void adj__accelerometer_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cacc_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<3, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::int32 & adj_sensordim,
    wp::vec_t<3, wp::float32> & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1541
static CUDA_CALLABLE void adj__force_0(
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cfrc_int_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1558
static CUDA_CALLABLE void adj__torque_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cfrc_int_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1579
static CUDA_CALLABLE void adj__actuator_force_0(
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_actuator_force_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1584
static CUDA_CALLABLE void adj__joint_actuator_force_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_dofadr,
    wp::array_t<wp::float32> & adj_qfrc_actuator_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1677
static CUDA_CALLABLE void adj__framelinacc_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cacc_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1728
static CUDA_CALLABLE void adj__frameangacc_0(
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cacc_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::vec_t<3, wp::float32> & adj_ret)
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


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:161
static CUDA_CALLABLE void adj_quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:177
static CUDA_CALLABLE void adj_quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb,
    wp::quat_t<wp::float32> & adj_qa,
    wp::quat_t<wp::float32> & adj_qb,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:322
static CUDA_CALLABLE void adj_upper_tri_index_0(
    wp::int32 var_n,
    wp::int32 var_i,
    wp::int32 var_j,
    wp::int32 & adj_n,
    wp::int32 & adj_i,
    wp::int32 & adj_j,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:675
static CUDA_CALLABLE void adj_inside_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_point,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_geomtype,
    wp::vec_t<3, wp::float32> & adj_point,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:2311
static CUDA_CALLABLE void adj__check_match_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::int32 var_body,
    wp::int32 var_geom,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::int32 & adj_body,
    wp::int32 & adj_geom,
    wp::int32 & adj_objtype,
    wp::int32 & adj_objid,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:963
static CUDA_CALLABLE void adj__velocimeter_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:988
static CUDA_CALLABLE void adj__gyro_0(
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1006
static CUDA_CALLABLE void adj__joint_vel_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_dofadr,
    wp::array_t<wp::float32> & adj_qvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1011
static CUDA_CALLABLE void adj__tendon_vel_0(
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_ten_velocity_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1016
static CUDA_CALLABLE void adj__actuator_vel_0(
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_actuator_velocity_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1021
static CUDA_CALLABLE void adj__ball_ang_vel_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_dofadr,
    wp::array_t<wp::float32> & adj_qvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1065
static CUDA_CALLABLE void adj__cvel_offset_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::vec_t<6, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_objid,
    wp::vec_t<6, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1107
static CUDA_CALLABLE void adj__frame_linvel_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1213
static CUDA_CALLABLE void adj__frame_angvel_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1295
static CUDA_CALLABLE void adj__subtree_linvel_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_linvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:1300
static CUDA_CALLABLE void adj__subtree_angmom_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_angmom_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:116
static CUDA_CALLABLE void adj__magnetometer_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_magnetic,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_opt_magnetic,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:130
static CUDA_CALLABLE void adj__cam_projection_0(
    wp::array_t<wp::float32> var_cam_fovy,
    wp::array_t<wp::vec_t<2, wp::int32>> var_cam_resolution,
    wp::array_t<wp::vec_t<2, wp::float32>> var_cam_sensorsize,
    wp::array_t<wp::vec_t<4, wp::float32>> var_cam_intrinsic,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_refid,
    wp::array_t<wp::float32> & adj_cam_fovy,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_cam_resolution,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_cam_sensorsize,
    wp::array_t<wp::vec_t<4, wp::float32>> & adj_cam_intrinsic,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_refid,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<2, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::int32 & adj_sensordim,
    wp::vec_t<2, wp::float32> & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:200
static CUDA_CALLABLE void adj__joint_pos_0(
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_qposadr,
    wp::array_t<wp::float32> & adj_qpos_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:205
static CUDA_CALLABLE void adj__tendon_pos_0(
    wp::array_t<wp::float32> var_ten_length_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_ten_length_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:210
static CUDA_CALLABLE void adj__actuator_pos_0(
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_actuator_length_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:215
static CUDA_CALLABLE void adj__ball_quat_0(
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_qposadr,
    wp::array_t<wp::float32> & adj_qpos_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::quat_t<wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::int32 & adj_sensordim,
    wp::quat_t<wp::float32> & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:265
static CUDA_CALLABLE void adj__get_pos_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:292
static CUDA_CALLABLE void adj__get_mat_0(
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_ximat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_objid,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:376
static CUDA_CALLABLE void adj__frame_pos_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:405
static CUDA_CALLABLE void adj__frame_axis_0(
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::int32 var_frame_axis,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_ximat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::int32 & adj_frame_axis,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:341
static CUDA_CALLABLE void adj__get_quat_0(
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_body_iquat,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_geom_quat,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_site_quat,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_cam_quat,
    wp::array_t<wp::quat_t<wp::float32>> & adj_xquat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_objid,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE void adj_quat_inv_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:431
static CUDA_CALLABLE void adj__frame_quat_0(
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::array_t<wp::quat_t<wp::float32>> & adj_body_iquat,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_geom_quat,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_site_quat,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_cam_quat,
    wp::array_t<wp::quat_t<wp::float32>> & adj_xquat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:484
static CUDA_CALLABLE void adj__subtree_com_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<6, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::int32 & adj_sensordim,
    wp::vec_t<6, wp::float32> & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /opt/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/sensor.py:489
static CUDA_CALLABLE void adj__clock_0(
    wp::array_t<wp::float32> var_time_in,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> & adj_time_in,
    wp::int32 & adj_worldid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _sensor_touch_0008103e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_type,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_size,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::int32> var_sensor_touch_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        const wp::int32 var_2 = 0;
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::vec_t<2, wp::int32>* var_15;
        wp::vec_t<2, wp::int32> var_16;
        wp::vec_t<2, wp::int32> var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::int32* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::vec_t<2, wp::int32> var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        bool var_34;
        const wp::int32 var_35 = 0;
        bool var_36;
        bool var_37;
        const wp::int32 var_38 = 0;
        wp::int32 var_39;
        bool var_40;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        bool var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 2;
        const wp::int32 var_53 = 1;
        wp::int32 var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 1;
        wp::range_t var_57;
        wp::int32 var_58;
        wp::int32* var_59;
        wp::float32* var_60;
        wp::int32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::float32 var_64 = 0.0;
        bool var_65;
        wp::mat_t<3, 3, wp::float32>* var_66;
        wp::mat_t<3, 3, wp::float32> var_67;
        wp::mat_t<3, 3, wp::float32> var_68;
        const wp::int32 var_69 = 0;
        const wp::int32 var_70 = 0;
        wp::float32 var_71;
        const wp::int32 var_72 = 0;
        const wp::int32 var_73 = 1;
        wp::float32 var_74;
        const wp::int32 var_75 = 0;
        const wp::int32 var_76 = 2;
        wp::float32 var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::float32 var_81;
        const wp::int32 var_82 = 1;
        wp::int32 var_83;
        bool var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32>* var_87;
        wp::mat_t<3, 3, wp::float32>* var_88;
        wp::vec_t<3, wp::float32>* var_89;
        wp::vec_t<3, wp::float32>* var_90;
        wp::int32* var_91;
        wp::float32 var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::mat_t<3, 3, wp::float32> var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::int32 var_98;
        const wp::float32 var_99 = 0.0;
        bool var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::array_t<wp::float32> var_104;
        wp::float32 var_105;
        //---------
        // forward
        // def _sensor_touch(                                                                     <L 2060>
        // conid, sensortouchadrid = wp.tid()                                                     <L 2084>
        builtin_tid2d(var_0, var_1);
        // if conid >= nacon_in[0]:                                                               <L 2086>
        var_3 = wp::address(var_nacon_in, var_2);
        var_5 = wp::load(var_3);
        var_4 = (var_0 >= var_5);
        if (var_4) {
            // return                                                                             <L 2087>
            return;
        }
        // sensorid = sensor_touch_adr[sensortouchadrid]                                          <L 2089>
        var_6 = wp::address(var_sensor_touch_adr, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // objid = sensor_objid[sensorid]                                                         <L 2091>
        var_9 = wp::address(var_sensor_objid, var_7);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // bodyid = site_bodyid[objid]                                                            <L 2092>
        var_12 = wp::address(var_site_bodyid, var_10);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // geom = contact_geom_in[conid]                                                          <L 2097>
        var_15 = wp::address(var_contact_geom_in, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // conbody = wp.vec2i(geom_bodyid[geom[0]], geom_bodyid[geom[1]])                         <L 2098>
        var_19 = wp::extract(var_16, var_18);
        var_20 = wp::address(var_geom_bodyid, var_19);
        var_22 = wp::extract(var_16, var_21);
        var_23 = wp::address(var_geom_bodyid, var_22);
        var_25 = wp::load(var_20);
        var_26 = wp::load(var_23);
        var_24 = wp::vec_t<2, wp::int32>(var_25, var_26);
        // worldid = contact_worldid_in[conid]                                                    <L 2101>
        var_27 = wp::address(var_contact_worldid_in, var_0);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // efc_address0 = contact_efc_address_in[conid, 0]                                        <L 2102>
        var_31 = wp::address(var_contact_efc_address_in, var_0, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // if efc_address0 >= 0 and (bodyid == conbody[0] or bodyid == conbody[1]):               <L 2103>
        var_36 = (var_32 >= var_35);
        var_34 = var_36;
        if (var_34) {
            var_39 = wp::extract(var_24, var_38);
            var_40 = (var_13 == var_39);
            var_37 = var_40;
            if (!var_37) {
                var_42 = wp::extract(var_24, var_41);
                var_43 = (var_13 == var_42);
                var_37 = var_37 || var_43;
            }
            var_34 = var_34 && var_37;
        }
        if (var_34) {
            // normalforce = efc_force_in[worldid, efc_address0]                                  <L 2105>
            var_44 = wp::address(var_efc_force_in, var_28, var_32);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // if opt_cone == ConeType.PYRAMIDAL:                                                 <L 2107>
            var_48 = (var_opt_cone == var_47);
            if (var_48) {
                // dim = contact_dim_in[conid]                                                    <L 2108>
                var_49 = wp::address(var_contact_dim_in, var_0);
                var_51 = wp::load(var_49);
                var_50 = wp::copy(var_51);
                // for i in range(1, 2 * (dim - 1)):                                              <L 2109>
                var_54 = wp::sub(var_50, var_53);
                var_55 = wp::mul(var_52, var_54);
                var_57 = wp::range(var_56, var_55);
                start_for_1:;
                    if (iter_cmp(var_57) == 0) goto end_for_1;
                    var_58 = wp::iter_next(var_57);
                    // normalforce += efc_force_in[worldid, contact_efc_address_in[conid, i]]       <L 2110>
                    var_59 = wp::address(var_contact_efc_address_in, var_0, var_58);
                    var_61 = wp::load(var_59);
                    var_60 = wp::address(var_efc_force_in, var_28, var_61);
                    var_63 = wp::load(var_60);
                    var_62 = wp::add(var_45, var_63);
                    wp::assign(var_45, var_62);
                    goto start_for_1;
                end_for_1:;
            }
            // if normalforce <= 0.0:                                                             <L 2112>
            var_65 = (var_45 <= var_64);
            if (var_65) {
                // return                                                                         <L 2113>
                return;
            }
            // frame = contact_frame_in[conid]                                                    <L 2116>
            var_66 = wp::address(var_contact_frame_in, var_0);
            var_68 = wp::load(var_66);
            var_67 = wp::copy(var_68);
            // conray = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2]) * normalforce              <L 2117>
            var_71 = wp::extract(var_67, var_69, var_70);
            var_74 = wp::extract(var_67, var_72, var_73);
            var_77 = wp::extract(var_67, var_75, var_76);
            var_78 = wp::vec_t<3, wp::float32>(var_71, var_74, var_77);
            var_79 = wp::mul(var_78, var_45);
            // conray, _ = math.normalize_with_norm(conray)                                       <L 2118>
            normalize_with_norm_1(var_79, var_80, var_81);
            // if bodyid == conbody[1]:                                                           <L 2121>
            var_83 = wp::extract(var_24, var_82);
            var_84 = (var_13 == var_83);
            if (var_84) {
                // conray = -conray                                                               <L 2122>
                var_85 = wp::neg(var_80);
            }
            var_86 = wp::where(var_84, var_85, var_80);
            // dist, normal = ray.ray_geom(                                                       <L 2125>
            // site_xpos_in[worldid, objid],                                                      <L 2126>
            var_87 = wp::address(var_site_xpos_in, var_28, var_10);
            // site_xmat_in[worldid, objid],                                                      <L 2127>
            var_88 = wp::address(var_site_xmat_in, var_28, var_10);
            // site_size[objid],                                                                  <L 2128>
            var_89 = wp::address(var_site_size, var_10);
            // contact_pos_in[conid],                                                             <L 2129>
            var_90 = wp::address(var_contact_pos_in, var_0);
            // conray,                                                                            <L 2130>
            // site_type[objid],                                                                  <L 2131>
            var_91 = wp::address(var_site_type, var_10);
            var_94 = wp::load(var_87);
            var_95 = wp::load(var_88);
            var_96 = wp::load(var_89);
            var_97 = wp::load(var_90);
            var_98 = wp::load(var_91);
            ray_geom_0(var_94, var_95, var_96, var_97, var_86, var_98, var_92, var_93);
            // if dist >= 0.0:                                                                    <L 2133>
            var_100 = (var_92 >= var_99);
            if (var_100) {
                // adr = sensor_adr[sensorid]                                                     <L 2134>
                var_101 = wp::address(var_sensor_adr, var_7);
                var_103 = wp::load(var_101);
                var_102 = wp::copy(var_103);
                // wp.atomic_add(sensordata_out[worldid], adr, normalforce)                       <L 2135>
                var_104 = wp::view(var_sensordata_out, var_28);
                var_105 = wp::atomic_add(var_104, var_102, var_45);
            }
        }
}



extern "C" __global__ void _preprocess_tactile_contacts_103a55ca_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_weld_geom_count_out,
    wp::array_t<wp::int32> var_weld_geom_list_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32>* var_9;
        wp::vec_t<2, wp::int32> var_10;
        wp::vec_t<2, wp::int32> var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        const wp::int32 var_31 = 0;
        bool var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::array_t<wp::int32> var_39;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        const wp::int32 var_42 = 50;
        bool var_43;
        const wp::int32 var_44 = 1;
        const wp::int32 var_45 = 0;
        bool var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::array_t<wp::int32> var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        bool var_56;
        //---------
        // forward
        // def _preprocess_tactile_contacts(                                                      <L 2144>
        // conid = wp.tid()                                                                       <L 2156>
        var_0 = builtin_tid1d();
        // ncon = nacon_in[0]                                                                     <L 2157>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if conid >= ncon:                                                                      <L 2158>
        var_5 = (var_0 >= var_3);
        if (var_5) {
            // return                                                                             <L 2159>
            continue;
        }
        // worldid = contact_worldid_in[conid]                                                    <L 2160>
        var_6 = wp::address(var_contact_worldid_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // contact_geom = contact_geom_in[conid]                                                  <L 2161>
        var_9 = wp::address(var_contact_geom_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // weld1 = body_weldid[geom_bodyid[contact_geom[0]]]                                      <L 2162>
        var_13 = wp::extract(var_10, var_12);
        var_14 = wp::address(var_geom_bodyid, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::address(var_body_weldid, var_16);
        var_18 = wp::load(var_15);
        var_17 = wp::copy(var_18);
        // weld2 = body_weldid[geom_bodyid[contact_geom[1]]]                                      <L 2163>
        var_20 = wp::extract(var_10, var_19);
        var_21 = wp::address(var_geom_bodyid, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::address(var_body_weldid, var_23);
        var_25 = wp::load(var_22);
        var_24 = wp::copy(var_25);
        // geom1 = contact_geom[0]                                                                <L 2164>
        var_27 = wp::extract(var_10, var_26);
        // geom2 = contact_geom[1]                                                                <L 2165>
        var_29 = wp::extract(var_10, var_28);
        // for side in range(2):                                                                  <L 2167>
        // if side == 0:                                                                          <L 2168>
        var_32 = (var_30 == var_31);
        if (var_32) {
            // weld = weld1                                                                       <L 2169>
            var_33 = wp::copy(var_17);
            // geom = geom2                                                                       <L 2170>
            var_34 = wp::copy(var_29);
        }
        if (!var_32) {
            // weld = weld2                                                                       <L 2172>
            var_35 = wp::copy(var_24);
            // geom = geom1                                                                       <L 2173>
            var_36 = wp::copy(var_27);
        }
        var_37 = wp::where(var_32, var_33, var_35);
        var_38 = wp::where(var_32, var_34, var_36);
        // idx = wp.atomic_add(weld_geom_count_out[worldid], weld, 1)                             <L 2175>
        var_39 = wp::view(var_weld_geom_count_out, var_7);
        var_41 = wp::atomic_add(var_39, var_37, var_40);
        // if idx < MJ_MAXCONPAIR:                                                                <L 2176>
        var_43 = (var_41 < var_42);
        if (var_43) {
            // weld_geom_list_out[worldid, weld, idx] = geom                                      <L 2177>
            wp::array_store(var_weld_geom_list_out, var_7, var_37, var_41, var_38);
        }
        // if side == 0:                                                                          <L 2168>
        var_46 = (var_44 == var_45);
        if (var_46) {
            // weld = weld1                                                                       <L 2169>
            var_47 = wp::copy(var_17);
            // geom = geom2                                                                       <L 2170>
            var_48 = wp::copy(var_29);
        }
        if (!var_46) {
            // weld = weld2                                                                       <L 2172>
            var_49 = wp::copy(var_24);
            // geom = geom1                                                                       <L 2173>
            var_50 = wp::copy(var_27);
        }
        var_51 = wp::where(var_46, var_47, var_49);
        var_52 = wp::where(var_46, var_48, var_50);
        // idx = wp.atomic_add(weld_geom_count_out[worldid], weld, 1)                             <L 2175>
        var_53 = wp::view(var_weld_geom_count_out, var_7);
        var_55 = wp::atomic_add(var_53, var_51, var_54);
        // if idx < MJ_MAXCONPAIR:                                                                <L 2176>
        var_56 = (var_55 < var_42);
        if (var_56) {
            // weld_geom_list_out[worldid, weld, idx] = geom                                      <L 2177>
            wp::array_store(var_weld_geom_list_out, var_7, var_51, var_55, var_52);
        }
    }
}



extern "C" __global__ void _tendon_actuator_force_cutoff_1d17a2e3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_tendonactfrc_adr,
    wp::array_t<wp::float32> var_sensordata_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::array_t<wp::float32> var_11;
        //---------
        // forward
        // def _tendon_actuator_force_cutoff(                                                     <L 1619>
        // worldid, tenactfrcid = wp.tid()                                                        <L 1631>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_tendonactfrc_adr[tenactfrcid]                                        <L 1632>
        var_2 = wp::address(var_sensor_tendonactfrc_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // adr = sensor_adr[sensorid]                                                             <L 1633>
        var_5 = wp::address(var_sensor_adr, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // val = sensordata_in[worldid, adr]                                                      <L 1634>
        var_8 = wp::address(var_sensordata_in, var_0, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, sensordata_out[worldid])       <L 1636>
        var_11 = wp::view(var_sensordata_out, var_0);
        _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_9, var_11);
}



extern "C" __global__ void _energy_pos_passive_tendon_23ab7454_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_tendon_stiffness,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_stiffnesspoly,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_out)
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
        bool var_18;
        const wp::float32 var_19 = 0.0;
        bool var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        bool var_24;
        const wp::int32 var_25 = 1;
        wp::float32 var_26;
        const wp::float32 var_27 = 0.0;
        bool var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::shape_t* var_32;
        const wp::int32 var_33 = 0;
        wp::int32 var_34;
        wp::shape_t var_35;
        wp::int32 var_36;
        wp::vec_t<2, wp::float32>* var_37;
        wp::vec_t<2, wp::float32> var_38;
        wp::vec_t<2, wp::float32> var_39;
        const wp::int32 var_40 = 0;
        wp::float32 var_41;
        const wp::int32 var_42 = 1;
        wp::float32 var_43;
        bool var_44;
        wp::float32 var_45;
        bool var_46;
        wp::float32 var_47;
        const wp::float32 var_48 = 0.0;
        wp::float32 var_49;
        wp::float32 var_50;
        const wp::int32 var_51 = 0;
        wp::float32 var_52;
        const wp::float32 var_53 = 0.0;
        wp::vec_t<2, wp::float32> var_54;
        wp::vec_t<2, wp::float32> var_55;
        //---------
        // forward
        // def _energy_pos_passive_tendon(                                                        <L 2889>
        // worldid, tenid = wp.tid()                                                              <L 2899>
        builtin_tid2d(var_0, var_1);
        // tendon_stiffness_id = worldid % tendon_stiffness.shape[0]                              <L 2901>
        var_2 = &(var_tendon_stiffness.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // stiffness = tendon_stiffness[tendon_stiffness_id, tenid]                               <L 2902>
        var_7 = wp::address(var_tendon_stiffness, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]           <L 2903>
        var_10 = &(var_tendon_stiffnesspoly.shape);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::mod(var_0, var_12);
        var_15 = wp::address(var_tendon_stiffnesspoly, var_14, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // if stiffness == 0.0 and spoly[0] == 0.0 and spoly[1] == 0.0:                           <L 2905>
        var_20 = (var_8 == var_19);
        var_18 = var_20;
        if (var_18) {
            var_22 = wp::extract(var_16, var_21);
            var_24 = (var_22 == var_23);
            var_18 = var_18 && var_24;
        }
        if (var_18) {
            var_26 = wp::extract(var_16, var_25);
            var_28 = (var_26 == var_27);
            var_18 = var_18 && var_28;
        }
        if (var_18) {
            // return                                                                             <L 2906>
            return;
        }
        // length = ten_length_in[worldid, tenid]                                                 <L 2908>
        var_29 = wp::address(var_ten_length_in, var_0, var_1);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // tendon_lengthspring_id = worldid % tendon_lengthspring.shape[0]                        <L 2911>
        var_32 = &(var_tendon_lengthspring.shape);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        var_36 = wp::mod(var_0, var_34);
        // lengthspring = tendon_lengthspring[tendon_lengthspring_id, tenid]                      <L 2912>
        var_37 = wp::address(var_tendon_lengthspring, var_36, var_1);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // lower = lengthspring[0]                                                                <L 2913>
        var_41 = wp::extract(var_38, var_40);
        // upper = lengthspring[1]                                                                <L 2914>
        var_43 = wp::extract(var_38, var_42);
        // if length > upper:                                                                     <L 2916>
        var_44 = (var_30 > var_43);
        if (var_44) {
            // x = length - upper                                                                 <L 2917>
            var_45 = wp::sub(var_30, var_43);
        }
        if (!var_44) {
            // elif length < lower:                                                               <L 2918>
            var_46 = (var_30 < var_41);
            if (var_46) {
                // x = length - lower                                                             <L 2919>
                var_47 = wp::sub(var_30, var_41);
            }
            if (!var_46) {
                // x = 0.0                                                                        <L 2921>
            }
            var_49 = wp::where(var_46, var_47, var_48);
        }
        var_50 = wp::where(var_44, var_45, var_49);
        // energy = wp.vec2(poly_potential(stiffness, spoly, x, 0), 0.0)                          <L 2923>
        var_52 = poly_potential_0(var_8, var_16, var_50, var_51);
        var_54 = wp::vec_t<2, wp::float32>(var_52, var_53);
        // wp.atomic_add(energy_out, worldid, energy)                                             <L 2924>
        var_55 = wp::atomic_add(var_energy_out, var_0, var_54);
}



extern "C" __global__ void _energy_pos_gravity_2e5e6dd5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_gravity,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::float32 var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        wp::vec_t<2, wp::float32> var_24;
        wp::vec_t<2, wp::float32> var_25;
        //---------
        // forward
        // def _energy_pos_gravity(                                                               <L 2775>
        // worldid, bodyid = wp.tid()                                                             <L 2784>
        builtin_tid2d(var_0, var_1);
        // gravity = opt_gravity[worldid % opt_gravity.shape[0]]                                  <L 2785>
        var_2 = &(var_opt_gravity.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_gravity, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // bodyid += 1  # skip world body                                                         <L 2786>
        var_11 = wp::add(var_1, var_10);
        // energy = wp.vec2(                                                                      <L 2788>
        // body_mass[worldid % body_mass.shape[0], bodyid] * wp.dot(gravity, xipos_in[worldid, bodyid]),       <L 2789>
        var_12 = &(var_body_mass.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        var_17 = wp::address(var_body_mass, var_16, var_11);
        var_18 = wp::address(var_xipos_in, var_0, var_11);
        var_20 = wp::load(var_18);
        var_19 = wp::dot(var_8, var_20);
        var_22 = wp::load(var_17);
        var_21 = wp::mul(var_22, var_19);
        // 0.0,                                                                                   <L 2790>
        var_24 = wp::vec_t<2, wp::float32>(var_21, var_23);
        // wp.atomic_sub(energy_out, worldid, energy)                                             <L 2793>
        var_25 = wp::atomic_sub(var_energy_out, var_0, var_24);
}



extern "C" __global__ void _sensor_rangefinder_init_49b66287_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_rangefinder_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pnt_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vec_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::mat_t<3, 3, wp::float32>* var_11;
        wp::mat_t<3, 3, wp::float32> var_12;
        wp::mat_t<3, 3, wp::float32> var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 2;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        const wp::int32 var_18 = 2;
        wp::float32 var_19;
        const wp::int32 var_20 = 2;
        const wp::int32 var_21 = 2;
        wp::float32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        //---------
        // forward
        // def _sensor_rangefinder_init(                                                          <L 179>
        // worldid, rfid = wp.tid()                                                               <L 190>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_rangefinder_adr[rfid]                                                <L 191>
        var_2 = wp::address(var_sensor_rangefinder_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // objid = sensor_objid[sensorid]                                                         <L 192>
        var_5 = wp::address(var_sensor_objid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // site_xpos = site_xpos_in[worldid, objid]                                               <L 193>
        var_8 = wp::address(var_site_xpos_in, var_0, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // site_xmat = site_xmat_in[worldid, objid]                                               <L 194>
        var_11 = wp::address(var_site_xmat_in, var_0, var_6);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // pnt_out[worldid, rfid] = site_xpos                                                     <L 196>
        wp::array_store(var_pnt_out, var_0, var_1, var_9);
        // vec_out[worldid, rfid] = wp.vec3(site_xmat[0, 2], site_xmat[1, 2], site_xmat[2, 2])       <L 197>
        var_16 = wp::extract(var_12, var_14, var_15);
        var_19 = wp::extract(var_12, var_17, var_18);
        var_22 = wp::extract(var_12, var_20, var_21);
        var_23 = wp::vec_t<3, wp::float32>(var_16, var_19, var_22);
        wp::array_store(var_vec_out, var_0, var_1, var_23);
}



extern "C" __global__ void _sensor_tactile_4bf3c9e2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::float32>> var_oct_coeff,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_octadr,
    wp::array_t<wp::int32> var_mesh_normaladr,
    wp::array_t<wp::int32> var_mesh_normalnum,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_normal,
    wp::array_t<wp::quat_t<wp::float32>> var_mesh_quat,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_refid,
    wp::array_t<wp::int32> var_sensor_dim,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::int32> var_plugin,
    wp::array_t<wp::vec_t<128, wp::float32>> var_plugin_attr,
    wp::array_t<wp::int32> var_geom_plugin_index,
    wp::array_t<wp::int32> var_taxel_vertadr,
    wp::array_t<wp::int32> var_taxel_sensorid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::int32> var_weld_geom_count_in,
    wp::array_t<wp::int32> var_weld_geom_list_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        bool var_21;
        wp::int32* var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::mat_t<3, 3, wp::float32>* var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::mat_t<3, 3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::int32* var_39;
        const wp::int32 var_40 = 3;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        bool var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 3;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::quat_t<wp::float32>* var_53;
        wp::quat_t<wp::float32> var_54;
        wp::quat_t<wp::float32> var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        const wp::float32 var_59 = 0.0;
        const wp::float32 var_60 = 0.0;
        const wp::float32 var_61 = 0.0;
        wp::vec_t<3, wp::float32> var_62;
        const wp::float32 var_63 = 0.0;
        const wp::float32 var_64 = 0.0;
        const wp::float32 var_65 = 0.0;
        wp::vec_t<3, wp::float32> var_66;
        const wp::int32 var_67 = 1;
        wp::int32 var_68;
        wp::vec_t<3, wp::float32>* var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        const wp::int32 var_72 = 2;
        wp::int32 var_73;
        wp::vec_t<3, wp::float32>* var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        const wp::int32 var_79 = 50;
        wp::range_t var_80;
        wp::int32 var_81;
        bool var_82;
        wp::int32* var_83;
        wp::int32 var_84;
        wp::int32 var_85;
        const wp::int32 var_86 = 0;
        bool var_87;
        const wp::int32 var_88 = 0;
        wp::int32 var_89;
        wp::range_t var_90;
        wp::int32 var_91;
        wp::int32* var_92;
        bool var_93;
        wp::int32 var_94;
        const wp::int32 var_95 = 1;
        wp::int32 var_96;
        const wp::int32 var_97 = 1;
        wp::int32 var_98;
        bool var_99;
        wp::int32* var_100;
        wp::int32 var_101;
        wp::int32 var_102;
        wp::vec_t<3, wp::float32>* var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::mat_t<3, 3, wp::float32>* var_106;
        wp::mat_t<3, 3, wp::float32> var_107;
        wp::mat_t<3, 3, wp::float32> var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::int32* var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::shape_t* var_116;
        const wp::int32 var_117 = 0;
        wp::int32 var_118;
        wp::shape_t var_119;
        wp::int32 var_120;
        wp::vec_t<3, wp::float32>* var_121;
        wp::shape_t* var_122;
        const wp::int32 var_123 = 0;
        wp::int32 var_124;
        wp::shape_t var_125;
        wp::int32 var_126;
        wp::int32* var_127;
        wp::vec_t<128, wp::float32> var_128;
        wp::int32 var_129;
        VolumeData_53ac1a2d var_130;
        MeshData_52eaa0fa var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::int32 var_133;
        wp::float32 var_134;
        const wp::float32 var_135 = 0.0;
        wp::float32 var_136;
        const wp::float32 var_137 = 0.0;
        bool var_138;
        wp::vec_t<6, wp::float32>* var_139;
        wp::int32* var_140;
        wp::vec_t<3, wp::float32>* var_141;
        wp::int32 var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32> var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<6, wp::float32> var_146;
        wp::vec_t<6, wp::float32>* var_147;
        wp::vec_t<3, wp::float32>* var_148;
        wp::int32* var_149;
        wp::vec_t<3, wp::float32>* var_150;
        wp::int32 var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<6, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        const wp::float32 var_158 = 0.0;
        const wp::float32 var_159 = 0.0;
        const wp::float32 var_160 = 0.0;
        wp::vec_t<3, wp::float32> var_161;
        wp::float32 var_162;
        const wp::int32 var_163 = 0;
        wp::float32 var_164;
        wp::float32 var_165;
        const wp::int32 var_166 = 1;
        wp::float32 var_167;
        wp::float32 var_168;
        const wp::int32 var_169 = 2;
        wp::int32* var_170;
        const wp::int32 var_171 = 3;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::int32* var_174;
        const wp::int32 var_175 = 0;
        wp::int32 var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::int32 var_179;
        const wp::int32 var_180 = 0;
        wp::float32 var_181;
        wp::float32 var_182;
        wp::int32* var_183;
        const wp::int32 var_184 = 1;
        wp::int32 var_185;
        wp::int32 var_186;
        wp::int32 var_187;
        wp::int32 var_188;
        const wp::int32 var_189 = 1;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::int32* var_192;
        const wp::int32 var_193 = 2;
        wp::int32 var_194;
        wp::int32 var_195;
        wp::int32 var_196;
        wp::int32 var_197;
        const wp::int32 var_198 = 2;
        wp::float32 var_199;
        wp::float32 var_200;
        //---------
        // forward
        // def _sensor_tactile(                                                                   <L 2181>
        // worldid, taxelid = wp.tid()                                                            <L 2220>
        builtin_tid2d(var_0, var_1);
        // sensor_id = taxel_sensorid[taxelid]                                                    <L 2222>
        var_2 = wp::address(var_taxel_sensorid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // mesh_id = sensor_objid[sensor_id]                                                      <L 2223>
        var_5 = wp::address(var_sensor_objid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // geom_id = sensor_refid[sensor_id]                                                      <L 2224>
        var_8 = wp::address(var_sensor_refid, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // parent_body = geom_bodyid[geom_id]                                                     <L 2225>
        var_11 = wp::address(var_geom_bodyid, var_9);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // parent_weld = body_weldid[parent_body]                                                 <L 2226>
        var_14 = wp::address(var_body_weldid, var_12);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // geom_count = weld_geom_count_in[worldid, parent_weld]                                  <L 2228>
        var_17 = wp::address(var_weld_geom_count_in, var_0, var_15);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // if geom_count == 0:                                                                    <L 2229>
        var_21 = (var_18 == var_20);
        if (var_21) {
            // return                                                                             <L 2230>
            return;
        }
        // vertid = taxel_vertadr[taxelid] - mesh_vertadr[mesh_id]                                <L 2233>
        var_22 = wp::address(var_taxel_vertadr, var_1);
        var_23 = wp::address(var_mesh_vertadr, var_6);
        var_25 = wp::load(var_22);
        var_26 = wp::load(var_23);
        var_24 = wp::sub(var_25, var_26);
        // pos = mesh_vert[vertid + mesh_vertadr[mesh_id]]                                        <L 2234>
        var_27 = wp::address(var_mesh_vertadr, var_6);
        var_29 = wp::load(var_27);
        var_28 = wp::add(var_24, var_29);
        var_30 = wp::address(var_mesh_vert, var_28);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // xpos = geom_xmat_in[worldid, geom_id] @ pos                                            <L 2237>
        var_33 = wp::address(var_geom_xmat_in, var_0, var_9);
        var_35 = wp::load(var_33);
        var_34 = wp::mul(var_35, var_31);
        // xpos += geom_xpos_in[worldid, geom_id]                                                 <L 2238>
        var_36 = wp::address(var_geom_xpos_in, var_0, var_9);
        var_38 = wp::load(var_36);
        var_37 = wp::add(var_34, var_38);
        // has_frame = mesh_normalnum[mesh_id] == 3 * mesh_vertnum[mesh_id]                       <L 2240>
        var_39 = wp::address(var_mesh_normalnum, var_6);
        var_41 = wp::address(var_mesh_vertnum, var_6);
        var_43 = wp::load(var_41);
        var_42 = wp::mul(var_40, var_43);
        var_45 = wp::load(var_39);
        var_44 = (var_45 == var_42);
        // normal_stride = 3 if has_frame else 1                                                  <L 2241>
        if (var_44) {
        }
        if (!var_44) {
        }
        var_48 = wp::where(var_44, var_46, var_47);
        // offset = mesh_normaladr[mesh_id] + normal_stride * vertid                              <L 2242>
        var_49 = wp::address(var_mesh_normaladr, var_6);
        var_50 = wp::mul(var_48, var_24);
        var_52 = wp::load(var_49);
        var_51 = wp::add(var_52, var_50);
        // quat = mesh_quat[mesh_id]                                                              <L 2243>
        var_53 = wp::address(var_mesh_quat, var_6);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // normal = math.rot_vec_quat(mesh_normal[offset], quat)                                  <L 2244>
        var_56 = wp::address(var_mesh_normal, var_51);
        var_58 = wp::load(var_56);
        var_57 = rot_vec_quat_0(var_58, var_54);
        // tang1 = wp.vec3(0.0, 0.0, 0.0)                                                         <L 2245>
        var_62 = wp::vec_t<3, wp::float32>(var_59, var_60, var_61);
        // tang2 = wp.vec3(0.0, 0.0, 0.0)                                                         <L 2246>
        var_66 = wp::vec_t<3, wp::float32>(var_63, var_64, var_65);
        // if has_frame:                                                                          <L 2247>
        if (var_44) {
            // tang1 = math.rot_vec_quat(mesh_normal[offset + 1], quat)                           <L 2248>
            var_68 = wp::add(var_51, var_67);
            var_69 = wp::address(var_mesh_normal, var_68);
            var_71 = wp::load(var_69);
            var_70 = rot_vec_quat_0(var_71, var_54);
            // tang2 = math.rot_vec_quat(mesh_normal[offset + 2], quat)                           <L 2249>
            var_73 = wp::add(var_51, var_72);
            var_74 = wp::address(var_mesh_normal, var_73);
            var_76 = wp::load(var_74);
            var_75 = rot_vec_quat_0(var_76, var_54);
        }
        var_77 = wp::where(var_44, var_70, var_62);
        var_78 = wp::where(var_44, var_75, var_66);
        // for g in range(MJ_MAXCONPAIR):                                                         <L 2251>
        var_80 = wp::range(var_79);
        start_for_1:;
            if (iter_cmp(var_80) == 0) goto end_for_1;
            var_81 = wp::iter_next(var_80);
            // if g >= geom_count:                                                                <L 2252>
            var_82 = (var_81 >= var_18);
            if (var_82) {
                // break                                                                          <L 2253>
                goto end_for_1;
            }
            // geom = weld_geom_list_in[worldid, parent_weld, g]                                  <L 2255>
            var_83 = wp::address(var_weld_geom_list_in, var_0, var_15, var_81);
            var_85 = wp::load(var_83);
            var_84 = wp::copy(var_85);
            // if geom < 0:                                                                       <L 2256>
            var_87 = (var_84 < var_86);
            if (var_87) {
                // continue                                                                       <L 2257>
                goto start_for_1;
            }
            // is_dup = int(0)                                                                    <L 2259>
            var_89 = wp::int(var_88);
            // for j in range(g):                                                                 <L 2260>
            var_90 = wp::range(var_81);
            start_for_3:;
                if (iter_cmp(var_90) == 0) goto end_for_3;
                var_91 = wp::iter_next(var_90);
                // if weld_geom_list_in[worldid, parent_weld, j] == geom:                         <L 2261>
                var_92 = wp::address(var_weld_geom_list_in, var_0, var_15, var_91);
                var_94 = wp::load(var_92);
                var_93 = (var_94 == var_84);
                if (var_93) {
                    // is_dup = int(1)                                                            <L 2262>
                    var_96 = wp::int(var_95);
                    // break                                                                      <L 2263>
                    wp::assign(var_89, var_96);
                    goto end_for_3;
                }
                goto start_for_3;
            end_for_3:;
            // if is_dup == int(1):                                                               <L 2264>
            var_98 = wp::int(var_97);
            var_99 = (var_89 == var_98);
            if (var_99) {
                // continue                                                                       <L 2265>
                goto start_for_1;
            }
            // body = geom_bodyid[geom]                                                           <L 2267>
            var_100 = wp::address(var_geom_bodyid, var_84);
            var_102 = wp::load(var_100);
            var_101 = wp::copy(var_102);
            // tmp = xpos - geom_xpos_in[worldid, geom]                                           <L 2269>
            var_103 = wp::address(var_geom_xpos_in, var_0, var_84);
            var_105 = wp::load(var_103);
            var_104 = wp::sub(var_37, var_105);
            // lpos = wp.transpose(geom_xmat_in[worldid, geom]) @ tmp                             <L 2270>
            var_106 = wp::address(var_geom_xmat_in, var_0, var_84);
            var_108 = wp::load(var_106);
            var_107 = wp::transpose(var_108);
            var_109 = wp::mul(var_107, var_104);
            // plugin_id = geom_plugin_index[geom]                                                <L 2272>
            var_110 = wp::address(var_geom_plugin_index, var_84);
            var_112 = wp::load(var_110);
            var_111 = wp::copy(var_112);
            // contact_type = geom_type[geom]                                                     <L 2273>
            var_113 = wp::address(var_geom_type, var_84);
            var_115 = wp::load(var_113);
            var_114 = wp::copy(var_115);
            // plugin_attributes, plugin_index, volume_data, mesh_data = get_sdf_params(          <L 2275>
            // oct_child,                                                                         <L 2276>
            // oct_aabb,                                                                          <L 2277>
            // oct_coeff,                                                                         <L 2278>
            // mesh_octadr,                                                                       <L 2279>
            // plugin,                                                                            <L 2280>
            // plugin_attr,                                                                       <L 2281>
            // contact_type,                                                                      <L 2282>
            // geom_size[worldid % geom_size.shape[0], geom],                                     <L 2283>
            var_116 = &(var_geom_size.shape);
            var_119 = wp::load(var_116);
            var_118 = wp::extract(var_119, var_117);
            var_120 = wp::mod(var_0, var_118);
            var_121 = wp::address(var_geom_size, var_120, var_84);
            // plugin_id,                                                                         <L 2284>
            // geom_dataid[worldid % geom_dataid.shape[0], geom],                                 <L 2285>
            var_122 = &(var_geom_dataid.shape);
            var_125 = wp::load(var_122);
            var_124 = wp::extract(var_125, var_123);
            var_126 = wp::mod(var_0, var_124);
            var_127 = wp::address(var_geom_dataid, var_126, var_84);
            var_132 = wp::load(var_121);
            var_133 = wp::load(var_127);
            get_sdf_params_0(var_oct_child, var_oct_aabb, var_oct_coeff, var_mesh_octadr, var_plugin, var_plugin_attr, var_114, var_132, var_111, var_133, var_128, var_129, var_130, var_131);
            // depth = wp.min(sdf(contact_type, lpos, plugin_attributes, plugin_index, volume_data, mesh_data), 0.0)       <L 2288>
            var_134 = sdf_0(var_114, var_109, var_128, var_129, var_130, var_131);
            var_136 = wp::min(var_134, var_135);
            // if depth >= 0.0:                                                                   <L 2289>
            var_138 = (var_136 >= var_137);
            if (var_138) {
                // continue                                                                       <L 2290>
                goto start_for_1;
            }
            // vel_sensor = _transform_spatial(cvel_in[worldid, parent_weld], xpos - subtree_com_in[worldid, body_rootid[parent_weld]])       <L 2292>
            var_139 = wp::address(var_cvel_in, var_0, var_15);
            var_140 = wp::address(var_body_rootid, var_15);
            var_142 = wp::load(var_140);
            var_141 = wp::address(var_subtree_com_in, var_0, var_142);
            var_144 = wp::load(var_141);
            var_143 = wp::sub(var_37, var_144);
            var_146 = wp::load(var_139);
            var_145 = _transform_spatial_0(var_146, var_143);
            // vel_other = _transform_spatial(                                                    <L 2293>
            // cvel_in[worldid, body], geom_xpos_in[worldid, geom] - subtree_com_in[worldid, body_rootid[body]]       <L 2294>
            var_147 = wp::address(var_cvel_in, var_0, var_101);
            var_148 = wp::address(var_geom_xpos_in, var_0, var_84);
            var_149 = wp::address(var_body_rootid, var_101);
            var_151 = wp::load(var_149);
            var_150 = wp::address(var_subtree_com_in, var_0, var_151);
            var_153 = wp::load(var_148);
            var_154 = wp::load(var_150);
            var_152 = wp::sub(var_153, var_154);
            var_156 = wp::load(var_147);
            var_155 = _transform_spatial_0(var_156, var_152);
            // vel_rel = vel_sensor - vel_other                                                   <L 2296>
            var_157 = wp::sub(var_145, var_155);
            // forceT = wp.vec3(0.0, 0.0, 0.0)                                                    <L 2298>
            var_161 = wp::vec_t<3, wp::float32>(var_158, var_159, var_160);
            // forceT[0] = -depth                                                                 <L 2299>
            var_162 = wp::neg(var_136);
            wp::assign_inplace(var_161, var_163, var_162);
            // if has_frame:                                                                      <L 2301>
            if (var_44) {
                // forceT[1] = wp.abs(wp.dot(vel_rel, tang1))                                     <L 2302>
                var_164 = wp::dot(var_157, var_77);
                var_165 = wp::abs(var_164);
                wp::assign_inplace(var_161, var_166, var_165);
                // forceT[2] = wp.abs(wp.dot(vel_rel, tang2))                                     <L 2303>
                var_167 = wp::dot(var_157, var_78);
                var_168 = wp::abs(var_167);
                wp::assign_inplace(var_161, var_169, var_168);
            }
            // dim = sensor_dim[sensor_id] // 3                                                   <L 2305>
            var_170 = wp::address(var_sensor_dim, var_3);
            var_173 = wp::load(var_170);
            var_172 = wp::floordiv(var_173, var_171);
            // wp.atomic_max(sensordata_out, worldid, sensor_adr[sensor_id] + 0 * dim + vertid, forceT[0])       <L 2306>
            var_174 = wp::address(var_sensor_adr, var_3);
            var_176 = wp::mul(var_175, var_172);
            var_178 = wp::load(var_174);
            var_177 = wp::add(var_178, var_176);
            var_179 = wp::add(var_177, var_24);
            var_181 = wp::extract(var_161, var_180);
            var_182 = wp::atomic_max(var_sensordata_out, var_0, var_179, var_181);
            // wp.atomic_add(sensordata_out, worldid, sensor_adr[sensor_id] + 1 * dim + vertid, forceT[1])       <L 2307>
            var_183 = wp::address(var_sensor_adr, var_3);
            var_185 = wp::mul(var_184, var_172);
            var_187 = wp::load(var_183);
            var_186 = wp::add(var_187, var_185);
            var_188 = wp::add(var_186, var_24);
            var_190 = wp::extract(var_161, var_189);
            var_191 = wp::atomic_add(var_sensordata_out, var_0, var_188, var_190);
            // wp.atomic_add(sensordata_out, worldid, sensor_adr[sensor_id] + 2 * dim + vertid, forceT[2])       <L 2308>
            var_192 = wp::address(var_sensor_adr, var_3);
            var_194 = wp::mul(var_193, var_172);
            var_196 = wp::load(var_192);
            var_195 = wp::add(var_196, var_194);
            var_197 = wp::add(var_195, var_24);
            var_199 = wp::extract(var_161, var_198);
            var_200 = wp::atomic_add(var_sensordata_out, var_0, var_197, var_199);
            goto start_for_1;
        end_for_1:;
}



extern "C" __global__ void _energy_pos_zero_5de26636_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        const wp::int32 var_2 = 0;
        //---------
        // forward
        // def _energy_pos_zero(                                                                  <L 2766>
        // worldid = wp.tid()                                                                     <L 2770>
        var_0 = builtin_tid1d();
        // energy_out[worldid][0] = 0.0                                                           <L 2771>
        wp::index(var_energy_out, var_0)[var_2] = var_1;
}



extern "C" __global__ void _sensor_acc_683a4930_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objtype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_intprm,
    wp::array_t<wp::int32> var_sensor_dim,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_acc_adr,
    wp::array_t<wp::int32> var_sensor_adr_to_contact_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_sensor_contact_nmatch_in,
    wp::array_t<wp::int32> var_sensor_contact_matchid_in,
    wp::array_t<wp::float32> var_sensor_contact_direction_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::array_t<wp::float32> var_11;
        const wp::int32 var_12 = 42;
        bool var_13;
        const wp::int32 var_14 = 0;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 1;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        const bool var_28 = false;
        const bool var_29 = false;
        const bool var_30 = false;
        const bool var_31 = false;
        const bool var_32 = false;
        const bool var_33 = false;
        const bool var_34 = false;
        const wp::int32 var_35 = 0;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::int32 var_41 = 0;
        bool var_42;
        const bool var_43 = true;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        const wp::int32 var_46 = 1;
        bool var_47;
        const bool var_48 = true;
        const wp::int32 var_49 = 3;
        wp::int32 var_50;
        const wp::int32 var_51 = 2;
        bool var_52;
        const bool var_53 = true;
        const wp::int32 var_54 = 3;
        wp::int32 var_55;
        const wp::int32 var_56 = 3;
        bool var_57;
        const bool var_58 = true;
        const wp::int32 var_59 = 1;
        wp::int32 var_60;
        const wp::int32 var_61 = 4;
        bool var_62;
        const bool var_63 = true;
        const wp::int32 var_64 = 3;
        wp::int32 var_65;
        const wp::int32 var_66 = 5;
        bool var_67;
        const bool var_68 = true;
        const wp::int32 var_69 = 3;
        wp::int32 var_70;
        const wp::int32 var_71 = 6;
        bool var_72;
        const bool var_73 = true;
        const wp::int32 var_74 = 3;
        wp::int32 var_75;
        bool var_76;
        wp::int32 var_77;
        bool var_78;
        bool var_79;
        wp::int32 var_80;
        bool var_81;
        bool var_82;
        bool var_83;
        wp::int32 var_84;
        bool var_85;
        bool var_86;
        bool var_87;
        bool var_88;
        wp::int32 var_89;
        bool var_90;
        bool var_91;
        bool var_92;
        bool var_93;
        bool var_94;
        wp::int32 var_95;
        bool var_96;
        bool var_97;
        bool var_98;
        bool var_99;
        bool var_100;
        bool var_101;
        wp::int32 var_102;
        bool var_103;
        bool var_104;
        bool var_105;
        bool var_106;
        bool var_107;
        bool var_108;
        bool var_109;
        wp::int32 var_110;
        bool var_111;
        bool var_112;
        bool var_113;
        bool var_114;
        bool var_115;
        bool var_116;
        bool var_117;
        wp::int32 var_118;
        const wp::int32 var_119 = 1;
        const wp::int32 var_120 = 1;
        wp::int32 var_121;
        wp::int32 var_122;
        const wp::int32 var_123 = 0;
        bool var_124;
        const bool var_125 = true;
        const wp::int32 var_126 = 1;
        wp::int32 var_127;
        const wp::int32 var_128 = 1;
        bool var_129;
        const bool var_130 = true;
        const wp::int32 var_131 = 3;
        wp::int32 var_132;
        const wp::int32 var_133 = 2;
        bool var_134;
        const bool var_135 = true;
        const wp::int32 var_136 = 3;
        wp::int32 var_137;
        const wp::int32 var_138 = 3;
        bool var_139;
        const bool var_140 = true;
        const wp::int32 var_141 = 1;
        wp::int32 var_142;
        const wp::int32 var_143 = 4;
        bool var_144;
        const bool var_145 = true;
        const wp::int32 var_146 = 3;
        wp::int32 var_147;
        const wp::int32 var_148 = 5;
        bool var_149;
        const bool var_150 = true;
        const wp::int32 var_151 = 3;
        wp::int32 var_152;
        const wp::int32 var_153 = 6;
        bool var_154;
        const bool var_155 = true;
        const wp::int32 var_156 = 3;
        wp::int32 var_157;
        bool var_158;
        wp::int32 var_159;
        bool var_160;
        bool var_161;
        wp::int32 var_162;
        bool var_163;
        bool var_164;
        bool var_165;
        wp::int32 var_166;
        bool var_167;
        bool var_168;
        bool var_169;
        bool var_170;
        wp::int32 var_171;
        bool var_172;
        bool var_173;
        bool var_174;
        bool var_175;
        bool var_176;
        wp::int32 var_177;
        bool var_178;
        bool var_179;
        bool var_180;
        bool var_181;
        bool var_182;
        bool var_183;
        wp::int32 var_184;
        bool var_185;
        bool var_186;
        bool var_187;
        bool var_188;
        bool var_189;
        bool var_190;
        bool var_191;
        wp::int32 var_192;
        bool var_193;
        bool var_194;
        bool var_195;
        bool var_196;
        bool var_197;
        bool var_198;
        bool var_199;
        wp::int32 var_200;
        const wp::int32 var_201 = 2;
        const wp::int32 var_202 = 1;
        wp::int32 var_203;
        wp::int32 var_204;
        const wp::int32 var_205 = 0;
        bool var_206;
        const bool var_207 = true;
        const wp::int32 var_208 = 1;
        wp::int32 var_209;
        const wp::int32 var_210 = 1;
        bool var_211;
        const bool var_212 = true;
        const wp::int32 var_213 = 3;
        wp::int32 var_214;
        const wp::int32 var_215 = 2;
        bool var_216;
        const bool var_217 = true;
        const wp::int32 var_218 = 3;
        wp::int32 var_219;
        const wp::int32 var_220 = 3;
        bool var_221;
        const bool var_222 = true;
        const wp::int32 var_223 = 1;
        wp::int32 var_224;
        const wp::int32 var_225 = 4;
        bool var_226;
        const bool var_227 = true;
        const wp::int32 var_228 = 3;
        wp::int32 var_229;
        const wp::int32 var_230 = 5;
        bool var_231;
        const bool var_232 = true;
        const wp::int32 var_233 = 3;
        wp::int32 var_234;
        const wp::int32 var_235 = 6;
        bool var_236;
        const bool var_237 = true;
        const wp::int32 var_238 = 3;
        wp::int32 var_239;
        bool var_240;
        wp::int32 var_241;
        bool var_242;
        bool var_243;
        wp::int32 var_244;
        bool var_245;
        bool var_246;
        bool var_247;
        wp::int32 var_248;
        bool var_249;
        bool var_250;
        bool var_251;
        bool var_252;
        wp::int32 var_253;
        bool var_254;
        bool var_255;
        bool var_256;
        bool var_257;
        bool var_258;
        wp::int32 var_259;
        bool var_260;
        bool var_261;
        bool var_262;
        bool var_263;
        bool var_264;
        bool var_265;
        wp::int32 var_266;
        bool var_267;
        bool var_268;
        bool var_269;
        bool var_270;
        bool var_271;
        bool var_272;
        bool var_273;
        wp::int32 var_274;
        bool var_275;
        bool var_276;
        bool var_277;
        bool var_278;
        bool var_279;
        bool var_280;
        bool var_281;
        wp::int32 var_282;
        const wp::int32 var_283 = 3;
        const wp::int32 var_284 = 1;
        wp::int32 var_285;
        wp::int32 var_286;
        const wp::int32 var_287 = 0;
        bool var_288;
        const bool var_289 = true;
        const wp::int32 var_290 = 1;
        wp::int32 var_291;
        const wp::int32 var_292 = 1;
        bool var_293;
        const bool var_294 = true;
        const wp::int32 var_295 = 3;
        wp::int32 var_296;
        const wp::int32 var_297 = 2;
        bool var_298;
        const bool var_299 = true;
        const wp::int32 var_300 = 3;
        wp::int32 var_301;
        const wp::int32 var_302 = 3;
        bool var_303;
        const bool var_304 = true;
        const wp::int32 var_305 = 1;
        wp::int32 var_306;
        const wp::int32 var_307 = 4;
        bool var_308;
        const bool var_309 = true;
        const wp::int32 var_310 = 3;
        wp::int32 var_311;
        const wp::int32 var_312 = 5;
        bool var_313;
        const bool var_314 = true;
        const wp::int32 var_315 = 3;
        wp::int32 var_316;
        const wp::int32 var_317 = 6;
        bool var_318;
        const bool var_319 = true;
        const wp::int32 var_320 = 3;
        wp::int32 var_321;
        bool var_322;
        wp::int32 var_323;
        bool var_324;
        bool var_325;
        wp::int32 var_326;
        bool var_327;
        bool var_328;
        bool var_329;
        wp::int32 var_330;
        bool var_331;
        bool var_332;
        bool var_333;
        bool var_334;
        wp::int32 var_335;
        bool var_336;
        bool var_337;
        bool var_338;
        bool var_339;
        bool var_340;
        wp::int32 var_341;
        bool var_342;
        bool var_343;
        bool var_344;
        bool var_345;
        bool var_346;
        bool var_347;
        wp::int32 var_348;
        bool var_349;
        bool var_350;
        bool var_351;
        bool var_352;
        bool var_353;
        bool var_354;
        bool var_355;
        wp::int32 var_356;
        bool var_357;
        bool var_358;
        bool var_359;
        bool var_360;
        bool var_361;
        bool var_362;
        bool var_363;
        wp::int32 var_364;
        const wp::int32 var_365 = 4;
        const wp::int32 var_366 = 1;
        wp::int32 var_367;
        wp::int32 var_368;
        const wp::int32 var_369 = 0;
        bool var_370;
        const bool var_371 = true;
        const wp::int32 var_372 = 1;
        wp::int32 var_373;
        const wp::int32 var_374 = 1;
        bool var_375;
        const bool var_376 = true;
        const wp::int32 var_377 = 3;
        wp::int32 var_378;
        const wp::int32 var_379 = 2;
        bool var_380;
        const bool var_381 = true;
        const wp::int32 var_382 = 3;
        wp::int32 var_383;
        const wp::int32 var_384 = 3;
        bool var_385;
        const bool var_386 = true;
        const wp::int32 var_387 = 1;
        wp::int32 var_388;
        const wp::int32 var_389 = 4;
        bool var_390;
        const bool var_391 = true;
        const wp::int32 var_392 = 3;
        wp::int32 var_393;
        const wp::int32 var_394 = 5;
        bool var_395;
        const bool var_396 = true;
        const wp::int32 var_397 = 3;
        wp::int32 var_398;
        const wp::int32 var_399 = 6;
        bool var_400;
        const bool var_401 = true;
        const wp::int32 var_402 = 3;
        wp::int32 var_403;
        bool var_404;
        wp::int32 var_405;
        bool var_406;
        bool var_407;
        wp::int32 var_408;
        bool var_409;
        bool var_410;
        bool var_411;
        wp::int32 var_412;
        bool var_413;
        bool var_414;
        bool var_415;
        bool var_416;
        wp::int32 var_417;
        bool var_418;
        bool var_419;
        bool var_420;
        bool var_421;
        bool var_422;
        wp::int32 var_423;
        bool var_424;
        bool var_425;
        bool var_426;
        bool var_427;
        bool var_428;
        bool var_429;
        wp::int32 var_430;
        bool var_431;
        bool var_432;
        bool var_433;
        bool var_434;
        bool var_435;
        bool var_436;
        bool var_437;
        wp::int32 var_438;
        bool var_439;
        bool var_440;
        bool var_441;
        bool var_442;
        bool var_443;
        bool var_444;
        bool var_445;
        wp::int32 var_446;
        const wp::int32 var_447 = 5;
        const wp::int32 var_448 = 1;
        wp::int32 var_449;
        wp::int32 var_450;
        const wp::int32 var_451 = 0;
        bool var_452;
        const bool var_453 = true;
        const wp::int32 var_454 = 1;
        wp::int32 var_455;
        const wp::int32 var_456 = 1;
        bool var_457;
        const bool var_458 = true;
        const wp::int32 var_459 = 3;
        wp::int32 var_460;
        const wp::int32 var_461 = 2;
        bool var_462;
        const bool var_463 = true;
        const wp::int32 var_464 = 3;
        wp::int32 var_465;
        const wp::int32 var_466 = 3;
        bool var_467;
        const bool var_468 = true;
        const wp::int32 var_469 = 1;
        wp::int32 var_470;
        const wp::int32 var_471 = 4;
        bool var_472;
        const bool var_473 = true;
        const wp::int32 var_474 = 3;
        wp::int32 var_475;
        const wp::int32 var_476 = 5;
        bool var_477;
        const bool var_478 = true;
        const wp::int32 var_479 = 3;
        wp::int32 var_480;
        const wp::int32 var_481 = 6;
        bool var_482;
        const bool var_483 = true;
        const wp::int32 var_484 = 3;
        wp::int32 var_485;
        bool var_486;
        wp::int32 var_487;
        bool var_488;
        bool var_489;
        wp::int32 var_490;
        bool var_491;
        bool var_492;
        bool var_493;
        wp::int32 var_494;
        bool var_495;
        bool var_496;
        bool var_497;
        bool var_498;
        wp::int32 var_499;
        bool var_500;
        bool var_501;
        bool var_502;
        bool var_503;
        bool var_504;
        wp::int32 var_505;
        bool var_506;
        bool var_507;
        bool var_508;
        bool var_509;
        bool var_510;
        bool var_511;
        wp::int32 var_512;
        bool var_513;
        bool var_514;
        bool var_515;
        bool var_516;
        bool var_517;
        bool var_518;
        bool var_519;
        wp::int32 var_520;
        bool var_521;
        bool var_522;
        bool var_523;
        bool var_524;
        bool var_525;
        bool var_526;
        bool var_527;
        wp::int32 var_528;
        const wp::int32 var_529 = 6;
        const wp::int32 var_530 = 1;
        wp::int32 var_531;
        wp::int32 var_532;
        const wp::int32 var_533 = 0;
        bool var_534;
        const bool var_535 = true;
        const wp::int32 var_536 = 1;
        wp::int32 var_537;
        const wp::int32 var_538 = 1;
        bool var_539;
        const bool var_540 = true;
        const wp::int32 var_541 = 3;
        wp::int32 var_542;
        const wp::int32 var_543 = 2;
        bool var_544;
        const bool var_545 = true;
        const wp::int32 var_546 = 3;
        wp::int32 var_547;
        const wp::int32 var_548 = 3;
        bool var_549;
        const bool var_550 = true;
        const wp::int32 var_551 = 1;
        wp::int32 var_552;
        const wp::int32 var_553 = 4;
        bool var_554;
        const bool var_555 = true;
        const wp::int32 var_556 = 3;
        wp::int32 var_557;
        const wp::int32 var_558 = 5;
        bool var_559;
        const bool var_560 = true;
        const wp::int32 var_561 = 3;
        wp::int32 var_562;
        const wp::int32 var_563 = 6;
        bool var_564;
        const bool var_565 = true;
        const wp::int32 var_566 = 3;
        wp::int32 var_567;
        bool var_568;
        wp::int32 var_569;
        bool var_570;
        bool var_571;
        wp::int32 var_572;
        bool var_573;
        bool var_574;
        bool var_575;
        wp::int32 var_576;
        bool var_577;
        bool var_578;
        bool var_579;
        bool var_580;
        wp::int32 var_581;
        bool var_582;
        bool var_583;
        bool var_584;
        bool var_585;
        bool var_586;
        wp::int32 var_587;
        bool var_588;
        bool var_589;
        bool var_590;
        bool var_591;
        bool var_592;
        bool var_593;
        wp::int32 var_594;
        bool var_595;
        bool var_596;
        bool var_597;
        bool var_598;
        bool var_599;
        bool var_600;
        bool var_601;
        wp::int32 var_602;
        bool var_603;
        bool var_604;
        bool var_605;
        bool var_606;
        bool var_607;
        bool var_608;
        bool var_609;
        wp::int32 var_610;
        wp::int32 var_611;
        wp::int32* var_612;
        wp::int32 var_613;
        wp::int32 var_614;
        wp::int32* var_615;
        wp::int32 var_616;
        wp::int32 var_617;
        wp::int32* var_618;
        wp::int32 var_619;
        wp::int32 var_620;
        const wp::int32 var_621 = 3;
        bool var_622;
        const wp::float32 var_623 = 0.0;
        wp::vec_t<3, wp::float32> var_624;
        const wp::float32 var_625 = 0.0;
        wp::vec_t<3, wp::float32> var_626;
        const wp::float32 var_627 = 0.0;
        wp::vec_t<3, wp::float32> var_628;
        const wp::float32 var_629 = 0.0;
        wp::float32 var_630;
        wp::range_t var_631;
        wp::int32 var_632;
        wp::int32* var_633;
        wp::int32 var_634;
        wp::int32 var_635;
        wp::float32* var_636;
        wp::float32 var_637;
        wp::float32 var_638;
        const bool var_639 = false;
        wp::vec_t<6, wp::float32> var_640;
        wp::vec_t<3, wp::float32> var_641;
        wp::float32 var_642;
        wp::vec_t<3, wp::float32>* var_643;
        wp::vec_t<3, wp::float32> var_644;
        wp::vec_t<3, wp::float32> var_645;
        wp::vec_t<3, wp::float32> var_646;
        wp::vec_t<3, wp::float32> var_647;
        wp::float32 var_648;
        wp::vec_t<6, wp::float32> var_649;
        wp::vec_t<3, wp::float32> var_650;
        wp::vec_t<3, wp::float32> var_651;
        wp::mat_t<3, 3, wp::float32>* var_652;
        wp::mat_t<3, 3, wp::float32> var_653;
        wp::mat_t<3, 3, wp::float32> var_654;
        wp::mat_t<3, 3, wp::float32> var_655;
        wp::vec_t<3, wp::float32> var_656;
        wp::vec_t<3, wp::float32> var_657;
        wp::vec_t<3, wp::float32> var_658;
        wp::vec_t<3, wp::float32> var_659;
        wp::vec_t<3, wp::float32> var_660;
        wp::vec_t<3, wp::float32> var_661;
        const wp::float32 var_662 = 1e-15;
        wp::float32 var_663;
        wp::vec_t<3, wp::float32> var_664;
        wp::vec_t<3, wp::float32> var_665;
        wp::vec_t<3, wp::float32> var_666;
        wp::int32 var_667;
        wp::float32 var_668;
        const wp::int32 var_669 = 1;
        wp::int32 var_670;
        wp::int32 var_671;
        const wp::int32 var_672 = 0;
        wp::float32 var_673;
        const wp::int32 var_674 = 0;
        wp::int32 var_675;
        const wp::int32 var_676 = 1;
        wp::float32 var_677;
        const wp::int32 var_678 = 1;
        wp::int32 var_679;
        const wp::int32 var_680 = 2;
        wp::float32 var_681;
        const wp::int32 var_682 = 2;
        wp::int32 var_683;
        const wp::int32 var_684 = 3;
        wp::int32 var_685;
        wp::int32 var_686;
        const wp::int32 var_687 = 0;
        wp::float32 var_688;
        const wp::int32 var_689 = 0;
        wp::int32 var_690;
        const wp::int32 var_691 = 1;
        wp::float32 var_692;
        const wp::int32 var_693 = 1;
        wp::int32 var_694;
        const wp::int32 var_695 = 2;
        wp::float32 var_696;
        const wp::int32 var_697 = 2;
        wp::int32 var_698;
        const wp::int32 var_699 = 3;
        wp::int32 var_700;
        wp::int32 var_701;
        const wp::float32 var_702 = 0.0;
        const wp::int32 var_703 = 1;
        wp::int32 var_704;
        wp::int32 var_705;
        const wp::int32 var_706 = 0;
        wp::float32 var_707;
        const wp::int32 var_708 = 0;
        wp::int32 var_709;
        const wp::int32 var_710 = 1;
        wp::float32 var_711;
        const wp::int32 var_712 = 1;
        wp::int32 var_713;
        const wp::int32 var_714 = 2;
        wp::float32 var_715;
        const wp::int32 var_716 = 2;
        wp::int32 var_717;
        const wp::int32 var_718 = 3;
        wp::int32 var_719;
        wp::int32 var_720;
        const wp::float32 var_721 = 1.0;
        const wp::int32 var_722 = 0;
        wp::int32 var_723;
        const wp::float32 var_724 = 0.0;
        const wp::int32 var_725 = 1;
        wp::int32 var_726;
        const wp::float32 var_727 = 0.0;
        const wp::int32 var_728 = 2;
        wp::int32 var_729;
        const wp::int32 var_730 = 3;
        wp::int32 var_731;
        wp::int32 var_732;
        const wp::float32 var_733 = 0.0;
        const wp::int32 var_734 = 0;
        wp::int32 var_735;
        const wp::float32 var_736 = 1.0;
        const wp::int32 var_737 = 1;
        wp::int32 var_738;
        const wp::float32 var_739 = 0.0;
        const wp::int32 var_740 = 2;
        wp::int32 var_741;
        wp::int32 var_742;
        wp::range_t var_743;
        wp::int32 var_744;
        wp::int32* var_745;
        wp::int32 var_746;
        wp::int32 var_747;
        wp::float32* var_748;
        wp::float32 var_749;
        wp::float32 var_750;
        wp::int32 var_751;
        wp::int32 var_752;
        wp::float32 var_753;
        const wp::int32 var_754 = 1;
        wp::int32 var_755;
        wp::int32 var_756;
        bool var_757;
        const bool var_758 = false;
        wp::vec_t<6, wp::float32> var_759;
        const wp::int32 var_760 = 0;
        wp::float32 var_761;
        const wp::int32 var_762 = 0;
        wp::int32 var_763;
        const wp::int32 var_764 = 1;
        wp::float32 var_765;
        const wp::int32 var_766 = 1;
        wp::int32 var_767;
        const wp::int32 var_768 = 2;
        wp::float32 var_769;
        wp::float32 var_770;
        const wp::int32 var_771 = 2;
        wp::int32 var_772;
        const wp::int32 var_773 = 3;
        wp::int32 var_774;
        wp::int32 var_775;
        const wp::int32 var_776 = 3;
        wp::float32 var_777;
        const wp::int32 var_778 = 0;
        wp::int32 var_779;
        const wp::int32 var_780 = 4;
        wp::float32 var_781;
        const wp::int32 var_782 = 1;
        wp::int32 var_783;
        const wp::int32 var_784 = 5;
        wp::float32 var_785;
        wp::float32 var_786;
        const wp::int32 var_787 = 2;
        wp::int32 var_788;
        const wp::int32 var_789 = 3;
        wp::int32 var_790;
        wp::int32 var_791;
        wp::float32* var_792;
        wp::float32 var_793;
        const wp::int32 var_794 = 1;
        wp::int32 var_795;
        wp::int32 var_796;
        wp::vec_t<3, wp::float32>* var_797;
        wp::vec_t<3, wp::float32> var_798;
        wp::vec_t<3, wp::float32> var_799;
        const wp::int32 var_800 = 0;
        wp::float32 var_801;
        const wp::int32 var_802 = 0;
        wp::int32 var_803;
        const wp::int32 var_804 = 1;
        wp::float32 var_805;
        const wp::int32 var_806 = 1;
        wp::int32 var_807;
        const wp::int32 var_808 = 2;
        wp::float32 var_809;
        const wp::int32 var_810 = 2;
        wp::int32 var_811;
        const wp::int32 var_812 = 3;
        wp::int32 var_813;
        wp::int32 var_814;
        wp::mat_t<3, 3, wp::float32>* var_815;
        const wp::int32 var_816 = 0;
        wp::vec_t<3, wp::float32> var_817;
        wp::mat_t<3, 3, wp::float32> var_818;
        const wp::int32 var_819 = 0;
        wp::float32 var_820;
        wp::float32 var_821;
        const wp::int32 var_822 = 0;
        wp::int32 var_823;
        const wp::int32 var_824 = 1;
        wp::float32 var_825;
        wp::float32 var_826;
        const wp::int32 var_827 = 1;
        wp::int32 var_828;
        const wp::int32 var_829 = 2;
        wp::float32 var_830;
        wp::float32 var_831;
        const wp::int32 var_832 = 2;
        wp::int32 var_833;
        const wp::int32 var_834 = 3;
        wp::int32 var_835;
        wp::int32 var_836;
        wp::mat_t<3, 3, wp::float32>* var_837;
        const wp::int32 var_838 = 1;
        wp::vec_t<3, wp::float32> var_839;
        wp::mat_t<3, 3, wp::float32> var_840;
        const wp::int32 var_841 = 0;
        wp::float32 var_842;
        wp::float32 var_843;
        const wp::int32 var_844 = 0;
        wp::int32 var_845;
        const wp::int32 var_846 = 1;
        wp::float32 var_847;
        wp::float32 var_848;
        const wp::int32 var_849 = 1;
        wp::int32 var_850;
        const wp::int32 var_851 = 2;
        wp::float32 var_852;
        wp::float32 var_853;
        const wp::int32 var_854 = 2;
        wp::int32 var_855;
        wp::range_t var_856;
        wp::int32 var_857;
        wp::range_t var_858;
        wp::int32 var_859;
        const wp::float32 var_860 = 0.0;
        wp::int32 var_861;
        wp::int32 var_862;
        wp::int32 var_863;
        wp::int32 var_864;
        wp::int32 var_865;
        wp::float32 var_866;
        wp::vec_t<6, wp::float32> var_867;
        wp::vec_t<3, wp::float32> var_868;
        wp::int32 var_869;
        const wp::int32 var_870 = 1;
        bool var_871;
        wp::vec_t<3, wp::float32> var_872;
        const wp::int32 var_873 = 3;
        const wp::int32 var_874 = 4;
        bool var_875;
        wp::vec_t<3, wp::float32> var_876;
        const wp::int32 var_877 = 3;
        const wp::int32 var_878 = 5;
        bool var_879;
        wp::vec_t<3, wp::float32> var_880;
        const wp::int32 var_881 = 3;
        const wp::int32 var_882 = 15;
        bool var_883;
        wp::float32 var_884;
        const wp::int32 var_885 = 16;
        bool var_886;
        wp::float32 var_887;
        const wp::int32 var_888 = 33;
        bool var_889;
        wp::int32* var_890;
        wp::int32 var_891;
        wp::int32 var_892;
        wp::vec_t<3, wp::float32> var_893;
        const wp::int32 var_894 = 3;
        const wp::int32 var_895 = 34;
        bool var_896;
        wp::int32* var_897;
        wp::int32 var_898;
        wp::int32 var_899;
        wp::vec_t<3, wp::float32> var_900;
        const wp::int32 var_901 = 3;
        wp::int32 var_902;
        wp::vec_t<3, wp::float32> var_903;
        wp::float32 var_904;
        wp::vec_t<3, wp::float32> var_905;
        wp::vec_t<3, wp::float32> var_906;
        wp::vec_t<3, wp::float32> var_907;
        wp::int32 var_908;
        //---------
        // forward
        // def _sensor_acc(                                                                       <L 1756>
        // worldid, accid = wp.tid()                                                              <L 1803>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_acc_adr[accid]                                                       <L 1804>
        var_2 = wp::address(var_sensor_acc_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // sensortype = sensor_type[sensorid]                                                     <L 1805>
        var_5 = wp::address(var_sensor_type, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // objid = sensor_objid[sensorid]                                                         <L 1806>
        var_8 = wp::address(var_sensor_objid, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // out = sensordata_out[worldid]                                                          <L 1807>
        var_11 = wp::view(var_sensordata_out, var_0);
        // if sensortype == SensorType.CONTACT:                                                   <L 1809>
        var_13 = (var_6 == var_12);
        if (var_13) {
            // dataspec = sensor_intprm[sensorid, 0]                                              <L 1810>
            var_15 = wp::address(var_sensor_intprm, var_3, var_14);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            // dim = sensor_dim[sensorid]                                                         <L 1811>
            var_18 = wp::address(var_sensor_dim, var_3);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // objtype = sensor_objtype[sensorid]                                                 <L 1812>
            var_21 = wp::address(var_sensor_objtype, var_3);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // reduce = sensor_intprm[sensorid, 1]                                                <L 1813>
            var_25 = wp::address(var_sensor_intprm, var_3, var_24);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            // found = False                                                                      <L 1817>
            // force = False                                                                      <L 1818>
            // torque = False                                                                     <L 1819>
            // dist = False                                                                       <L 1820>
            // pos = False                                                                        <L 1821>
            // normal = False                                                                     <L 1822>
            // tangent = False                                                                    <L 1823>
            // size = int(0)                                                                      <L 1825>
            var_36 = wp::int(var_35);
            // for i in range(7):                                                                 <L 1826>
            // if dataspec & (1 << i):                                                            <L 1827>
            var_39 = wp::lshift(var_38, var_37);
            var_40 = wp::bit_and(var_16, var_39);
            if (var_40) {
                // if i == 0:                                                                     <L 1828>
                var_42 = (var_37 == var_41);
                if (var_42) {
                    // found = True                                                               <L 1829>
                    // size += 1                                                                  <L 1830>
                    var_45 = wp::add(var_36, var_44);
                }
                if (!var_42) {
                    // elif i == 1:                                                               <L 1831>
                    var_47 = (var_37 == var_46);
                    if (var_47) {
                        // force = True                                                           <L 1832>
                        // size += 3                                                              <L 1833>
                        var_50 = wp::add(var_36, var_49);
                    }
                    if (!var_47) {
                        // elif i == 2:                                                           <L 1834>
                        var_52 = (var_37 == var_51);
                        if (var_52) {
                            // torque = True                                                      <L 1835>
                            // size += 3                                                          <L 1836>
                            var_55 = wp::add(var_36, var_54);
                        }
                        if (!var_52) {
                            // elif i == 3:                                                       <L 1837>
                            var_57 = (var_37 == var_56);
                            if (var_57) {
                                // dist = True                                                    <L 1838>
                                // size += 1                                                      <L 1839>
                                var_60 = wp::add(var_36, var_59);
                            }
                            if (!var_57) {
                                // elif i == 4:                                                   <L 1840>
                                var_62 = (var_37 == var_61);
                                if (var_62) {
                                    // pos = True                                                 <L 1841>
                                    // size += 3                                                  <L 1842>
                                    var_65 = wp::add(var_36, var_64);
                                }
                                if (!var_62) {
                                    // elif i == 5:                                               <L 1843>
                                    var_67 = (var_37 == var_66);
                                    if (var_67) {
                                        // normal = True                                          <L 1844>
                                        // size += 3                                              <L 1845>
                                        var_70 = wp::add(var_36, var_69);
                                    }
                                    if (!var_67) {
                                        // elif i == 6:                                           <L 1846>
                                        var_72 = (var_37 == var_71);
                                        if (var_72) {
                                            // tangent = True                                     <L 1847>
                                            // size += 3                                          <L 1848>
                                            var_75 = wp::add(var_36, var_74);
                                        }
                                        var_76 = wp::where(var_72, var_73, var_34);
                                        var_77 = wp::where(var_72, var_75, var_36);
                                    }
                                    var_78 = wp::where(var_67, var_68, var_33);
                                    var_79 = wp::where(var_67, var_34, var_76);
                                    var_80 = wp::where(var_67, var_70, var_77);
                                }
                                var_81 = wp::where(var_62, var_63, var_32);
                                var_82 = wp::where(var_62, var_33, var_78);
                                var_83 = wp::where(var_62, var_34, var_79);
                                var_84 = wp::where(var_62, var_65, var_80);
                            }
                            var_85 = wp::where(var_57, var_58, var_31);
                            var_86 = wp::where(var_57, var_32, var_81);
                            var_87 = wp::where(var_57, var_33, var_82);
                            var_88 = wp::where(var_57, var_34, var_83);
                            var_89 = wp::where(var_57, var_60, var_84);
                        }
                        var_90 = wp::where(var_52, var_53, var_30);
                        var_91 = wp::where(var_52, var_31, var_85);
                        var_92 = wp::where(var_52, var_32, var_86);
                        var_93 = wp::where(var_52, var_33, var_87);
                        var_94 = wp::where(var_52, var_34, var_88);
                        var_95 = wp::where(var_52, var_55, var_89);
                    }
                    var_96 = wp::where(var_47, var_48, var_29);
                    var_97 = wp::where(var_47, var_30, var_90);
                    var_98 = wp::where(var_47, var_31, var_91);
                    var_99 = wp::where(var_47, var_32, var_92);
                    var_100 = wp::where(var_47, var_33, var_93);
                    var_101 = wp::where(var_47, var_34, var_94);
                    var_102 = wp::where(var_47, var_50, var_95);
                }
                var_103 = wp::where(var_42, var_43, var_28);
                var_104 = wp::where(var_42, var_29, var_96);
                var_105 = wp::where(var_42, var_30, var_97);
                var_106 = wp::where(var_42, var_31, var_98);
                var_107 = wp::where(var_42, var_32, var_99);
                var_108 = wp::where(var_42, var_33, var_100);
                var_109 = wp::where(var_42, var_34, var_101);
                var_110 = wp::where(var_42, var_45, var_102);
            }
            var_111 = wp::where(var_40, var_103, var_28);
            var_112 = wp::where(var_40, var_104, var_29);
            var_113 = wp::where(var_40, var_105, var_30);
            var_114 = wp::where(var_40, var_106, var_31);
            var_115 = wp::where(var_40, var_107, var_32);
            var_116 = wp::where(var_40, var_108, var_33);
            var_117 = wp::where(var_40, var_109, var_34);
            var_118 = wp::where(var_40, var_110, var_36);
            // if dataspec & (1 << i):                                                            <L 1827>
            var_121 = wp::lshift(var_120, var_119);
            var_122 = wp::bit_and(var_16, var_121);
            if (var_122) {
                // if i == 0:                                                                     <L 1828>
                var_124 = (var_119 == var_123);
                if (var_124) {
                    // found = True                                                               <L 1829>
                    // size += 1                                                                  <L 1830>
                    var_127 = wp::add(var_118, var_126);
                }
                if (!var_124) {
                    // elif i == 1:                                                               <L 1831>
                    var_129 = (var_119 == var_128);
                    if (var_129) {
                        // force = True                                                           <L 1832>
                        // size += 3                                                              <L 1833>
                        var_132 = wp::add(var_118, var_131);
                    }
                    if (!var_129) {
                        // elif i == 2:                                                           <L 1834>
                        var_134 = (var_119 == var_133);
                        if (var_134) {
                            // torque = True                                                      <L 1835>
                            // size += 3                                                          <L 1836>
                            var_137 = wp::add(var_118, var_136);
                        }
                        if (!var_134) {
                            // elif i == 3:                                                       <L 1837>
                            var_139 = (var_119 == var_138);
                            if (var_139) {
                                // dist = True                                                    <L 1838>
                                // size += 1                                                      <L 1839>
                                var_142 = wp::add(var_118, var_141);
                            }
                            if (!var_139) {
                                // elif i == 4:                                                   <L 1840>
                                var_144 = (var_119 == var_143);
                                if (var_144) {
                                    // pos = True                                                 <L 1841>
                                    // size += 3                                                  <L 1842>
                                    var_147 = wp::add(var_118, var_146);
                                }
                                if (!var_144) {
                                    // elif i == 5:                                               <L 1843>
                                    var_149 = (var_119 == var_148);
                                    if (var_149) {
                                        // normal = True                                          <L 1844>
                                        // size += 3                                              <L 1845>
                                        var_152 = wp::add(var_118, var_151);
                                    }
                                    if (!var_149) {
                                        // elif i == 6:                                           <L 1846>
                                        var_154 = (var_119 == var_153);
                                        if (var_154) {
                                            // tangent = True                                     <L 1847>
                                            // size += 3                                          <L 1848>
                                            var_157 = wp::add(var_118, var_156);
                                        }
                                        var_158 = wp::where(var_154, var_155, var_117);
                                        var_159 = wp::where(var_154, var_157, var_118);
                                    }
                                    var_160 = wp::where(var_149, var_150, var_116);
                                    var_161 = wp::where(var_149, var_117, var_158);
                                    var_162 = wp::where(var_149, var_152, var_159);
                                }
                                var_163 = wp::where(var_144, var_145, var_115);
                                var_164 = wp::where(var_144, var_116, var_160);
                                var_165 = wp::where(var_144, var_117, var_161);
                                var_166 = wp::where(var_144, var_147, var_162);
                            }
                            var_167 = wp::where(var_139, var_140, var_114);
                            var_168 = wp::where(var_139, var_115, var_163);
                            var_169 = wp::where(var_139, var_116, var_164);
                            var_170 = wp::where(var_139, var_117, var_165);
                            var_171 = wp::where(var_139, var_142, var_166);
                        }
                        var_172 = wp::where(var_134, var_135, var_113);
                        var_173 = wp::where(var_134, var_114, var_167);
                        var_174 = wp::where(var_134, var_115, var_168);
                        var_175 = wp::where(var_134, var_116, var_169);
                        var_176 = wp::where(var_134, var_117, var_170);
                        var_177 = wp::where(var_134, var_137, var_171);
                    }
                    var_178 = wp::where(var_129, var_130, var_112);
                    var_179 = wp::where(var_129, var_113, var_172);
                    var_180 = wp::where(var_129, var_114, var_173);
                    var_181 = wp::where(var_129, var_115, var_174);
                    var_182 = wp::where(var_129, var_116, var_175);
                    var_183 = wp::where(var_129, var_117, var_176);
                    var_184 = wp::where(var_129, var_132, var_177);
                }
                var_185 = wp::where(var_124, var_125, var_111);
                var_186 = wp::where(var_124, var_112, var_178);
                var_187 = wp::where(var_124, var_113, var_179);
                var_188 = wp::where(var_124, var_114, var_180);
                var_189 = wp::where(var_124, var_115, var_181);
                var_190 = wp::where(var_124, var_116, var_182);
                var_191 = wp::where(var_124, var_117, var_183);
                var_192 = wp::where(var_124, var_127, var_184);
            }
            var_193 = wp::where(var_122, var_185, var_111);
            var_194 = wp::where(var_122, var_186, var_112);
            var_195 = wp::where(var_122, var_187, var_113);
            var_196 = wp::where(var_122, var_188, var_114);
            var_197 = wp::where(var_122, var_189, var_115);
            var_198 = wp::where(var_122, var_190, var_116);
            var_199 = wp::where(var_122, var_191, var_117);
            var_200 = wp::where(var_122, var_192, var_118);
            // if dataspec & (1 << i):                                                            <L 1827>
            var_203 = wp::lshift(var_202, var_201);
            var_204 = wp::bit_and(var_16, var_203);
            if (var_204) {
                // if i == 0:                                                                     <L 1828>
                var_206 = (var_201 == var_205);
                if (var_206) {
                    // found = True                                                               <L 1829>
                    // size += 1                                                                  <L 1830>
                    var_209 = wp::add(var_200, var_208);
                }
                if (!var_206) {
                    // elif i == 1:                                                               <L 1831>
                    var_211 = (var_201 == var_210);
                    if (var_211) {
                        // force = True                                                           <L 1832>
                        // size += 3                                                              <L 1833>
                        var_214 = wp::add(var_200, var_213);
                    }
                    if (!var_211) {
                        // elif i == 2:                                                           <L 1834>
                        var_216 = (var_201 == var_215);
                        if (var_216) {
                            // torque = True                                                      <L 1835>
                            // size += 3                                                          <L 1836>
                            var_219 = wp::add(var_200, var_218);
                        }
                        if (!var_216) {
                            // elif i == 3:                                                       <L 1837>
                            var_221 = (var_201 == var_220);
                            if (var_221) {
                                // dist = True                                                    <L 1838>
                                // size += 1                                                      <L 1839>
                                var_224 = wp::add(var_200, var_223);
                            }
                            if (!var_221) {
                                // elif i == 4:                                                   <L 1840>
                                var_226 = (var_201 == var_225);
                                if (var_226) {
                                    // pos = True                                                 <L 1841>
                                    // size += 3                                                  <L 1842>
                                    var_229 = wp::add(var_200, var_228);
                                }
                                if (!var_226) {
                                    // elif i == 5:                                               <L 1843>
                                    var_231 = (var_201 == var_230);
                                    if (var_231) {
                                        // normal = True                                          <L 1844>
                                        // size += 3                                              <L 1845>
                                        var_234 = wp::add(var_200, var_233);
                                    }
                                    if (!var_231) {
                                        // elif i == 6:                                           <L 1846>
                                        var_236 = (var_201 == var_235);
                                        if (var_236) {
                                            // tangent = True                                     <L 1847>
                                            // size += 3                                          <L 1848>
                                            var_239 = wp::add(var_200, var_238);
                                        }
                                        var_240 = wp::where(var_236, var_237, var_199);
                                        var_241 = wp::where(var_236, var_239, var_200);
                                    }
                                    var_242 = wp::where(var_231, var_232, var_198);
                                    var_243 = wp::where(var_231, var_199, var_240);
                                    var_244 = wp::where(var_231, var_234, var_241);
                                }
                                var_245 = wp::where(var_226, var_227, var_197);
                                var_246 = wp::where(var_226, var_198, var_242);
                                var_247 = wp::where(var_226, var_199, var_243);
                                var_248 = wp::where(var_226, var_229, var_244);
                            }
                            var_249 = wp::where(var_221, var_222, var_196);
                            var_250 = wp::where(var_221, var_197, var_245);
                            var_251 = wp::where(var_221, var_198, var_246);
                            var_252 = wp::where(var_221, var_199, var_247);
                            var_253 = wp::where(var_221, var_224, var_248);
                        }
                        var_254 = wp::where(var_216, var_217, var_195);
                        var_255 = wp::where(var_216, var_196, var_249);
                        var_256 = wp::where(var_216, var_197, var_250);
                        var_257 = wp::where(var_216, var_198, var_251);
                        var_258 = wp::where(var_216, var_199, var_252);
                        var_259 = wp::where(var_216, var_219, var_253);
                    }
                    var_260 = wp::where(var_211, var_212, var_194);
                    var_261 = wp::where(var_211, var_195, var_254);
                    var_262 = wp::where(var_211, var_196, var_255);
                    var_263 = wp::where(var_211, var_197, var_256);
                    var_264 = wp::where(var_211, var_198, var_257);
                    var_265 = wp::where(var_211, var_199, var_258);
                    var_266 = wp::where(var_211, var_214, var_259);
                }
                var_267 = wp::where(var_206, var_207, var_193);
                var_268 = wp::where(var_206, var_194, var_260);
                var_269 = wp::where(var_206, var_195, var_261);
                var_270 = wp::where(var_206, var_196, var_262);
                var_271 = wp::where(var_206, var_197, var_263);
                var_272 = wp::where(var_206, var_198, var_264);
                var_273 = wp::where(var_206, var_199, var_265);
                var_274 = wp::where(var_206, var_209, var_266);
            }
            var_275 = wp::where(var_204, var_267, var_193);
            var_276 = wp::where(var_204, var_268, var_194);
            var_277 = wp::where(var_204, var_269, var_195);
            var_278 = wp::where(var_204, var_270, var_196);
            var_279 = wp::where(var_204, var_271, var_197);
            var_280 = wp::where(var_204, var_272, var_198);
            var_281 = wp::where(var_204, var_273, var_199);
            var_282 = wp::where(var_204, var_274, var_200);
            // if dataspec & (1 << i):                                                            <L 1827>
            var_285 = wp::lshift(var_284, var_283);
            var_286 = wp::bit_and(var_16, var_285);
            if (var_286) {
                // if i == 0:                                                                     <L 1828>
                var_288 = (var_283 == var_287);
                if (var_288) {
                    // found = True                                                               <L 1829>
                    // size += 1                                                                  <L 1830>
                    var_291 = wp::add(var_282, var_290);
                }
                if (!var_288) {
                    // elif i == 1:                                                               <L 1831>
                    var_293 = (var_283 == var_292);
                    if (var_293) {
                        // force = True                                                           <L 1832>
                        // size += 3                                                              <L 1833>
                        var_296 = wp::add(var_282, var_295);
                    }
                    if (!var_293) {
                        // elif i == 2:                                                           <L 1834>
                        var_298 = (var_283 == var_297);
                        if (var_298) {
                            // torque = True                                                      <L 1835>
                            // size += 3                                                          <L 1836>
                            var_301 = wp::add(var_282, var_300);
                        }
                        if (!var_298) {
                            // elif i == 3:                                                       <L 1837>
                            var_303 = (var_283 == var_302);
                            if (var_303) {
                                // dist = True                                                    <L 1838>
                                // size += 1                                                      <L 1839>
                                var_306 = wp::add(var_282, var_305);
                            }
                            if (!var_303) {
                                // elif i == 4:                                                   <L 1840>
                                var_308 = (var_283 == var_307);
                                if (var_308) {
                                    // pos = True                                                 <L 1841>
                                    // size += 3                                                  <L 1842>
                                    var_311 = wp::add(var_282, var_310);
                                }
                                if (!var_308) {
                                    // elif i == 5:                                               <L 1843>
                                    var_313 = (var_283 == var_312);
                                    if (var_313) {
                                        // normal = True                                          <L 1844>
                                        // size += 3                                              <L 1845>
                                        var_316 = wp::add(var_282, var_315);
                                    }
                                    if (!var_313) {
                                        // elif i == 6:                                           <L 1846>
                                        var_318 = (var_283 == var_317);
                                        if (var_318) {
                                            // tangent = True                                     <L 1847>
                                            // size += 3                                          <L 1848>
                                            var_321 = wp::add(var_282, var_320);
                                        }
                                        var_322 = wp::where(var_318, var_319, var_281);
                                        var_323 = wp::where(var_318, var_321, var_282);
                                    }
                                    var_324 = wp::where(var_313, var_314, var_280);
                                    var_325 = wp::where(var_313, var_281, var_322);
                                    var_326 = wp::where(var_313, var_316, var_323);
                                }
                                var_327 = wp::where(var_308, var_309, var_279);
                                var_328 = wp::where(var_308, var_280, var_324);
                                var_329 = wp::where(var_308, var_281, var_325);
                                var_330 = wp::where(var_308, var_311, var_326);
                            }
                            var_331 = wp::where(var_303, var_304, var_278);
                            var_332 = wp::where(var_303, var_279, var_327);
                            var_333 = wp::where(var_303, var_280, var_328);
                            var_334 = wp::where(var_303, var_281, var_329);
                            var_335 = wp::where(var_303, var_306, var_330);
                        }
                        var_336 = wp::where(var_298, var_299, var_277);
                        var_337 = wp::where(var_298, var_278, var_331);
                        var_338 = wp::where(var_298, var_279, var_332);
                        var_339 = wp::where(var_298, var_280, var_333);
                        var_340 = wp::where(var_298, var_281, var_334);
                        var_341 = wp::where(var_298, var_301, var_335);
                    }
                    var_342 = wp::where(var_293, var_294, var_276);
                    var_343 = wp::where(var_293, var_277, var_336);
                    var_344 = wp::where(var_293, var_278, var_337);
                    var_345 = wp::where(var_293, var_279, var_338);
                    var_346 = wp::where(var_293, var_280, var_339);
                    var_347 = wp::where(var_293, var_281, var_340);
                    var_348 = wp::where(var_293, var_296, var_341);
                }
                var_349 = wp::where(var_288, var_289, var_275);
                var_350 = wp::where(var_288, var_276, var_342);
                var_351 = wp::where(var_288, var_277, var_343);
                var_352 = wp::where(var_288, var_278, var_344);
                var_353 = wp::where(var_288, var_279, var_345);
                var_354 = wp::where(var_288, var_280, var_346);
                var_355 = wp::where(var_288, var_281, var_347);
                var_356 = wp::where(var_288, var_291, var_348);
            }
            var_357 = wp::where(var_286, var_349, var_275);
            var_358 = wp::where(var_286, var_350, var_276);
            var_359 = wp::where(var_286, var_351, var_277);
            var_360 = wp::where(var_286, var_352, var_278);
            var_361 = wp::where(var_286, var_353, var_279);
            var_362 = wp::where(var_286, var_354, var_280);
            var_363 = wp::where(var_286, var_355, var_281);
            var_364 = wp::where(var_286, var_356, var_282);
            // if dataspec & (1 << i):                                                            <L 1827>
            var_367 = wp::lshift(var_366, var_365);
            var_368 = wp::bit_and(var_16, var_367);
            if (var_368) {
                // if i == 0:                                                                     <L 1828>
                var_370 = (var_365 == var_369);
                if (var_370) {
                    // found = True                                                               <L 1829>
                    // size += 1                                                                  <L 1830>
                    var_373 = wp::add(var_364, var_372);
                }
                if (!var_370) {
                    // elif i == 1:                                                               <L 1831>
                    var_375 = (var_365 == var_374);
                    if (var_375) {
                        // force = True                                                           <L 1832>
                        // size += 3                                                              <L 1833>
                        var_378 = wp::add(var_364, var_377);
                    }
                    if (!var_375) {
                        // elif i == 2:                                                           <L 1834>
                        var_380 = (var_365 == var_379);
                        if (var_380) {
                            // torque = True                                                      <L 1835>
                            // size += 3                                                          <L 1836>
                            var_383 = wp::add(var_364, var_382);
                        }
                        if (!var_380) {
                            // elif i == 3:                                                       <L 1837>
                            var_385 = (var_365 == var_384);
                            if (var_385) {
                                // dist = True                                                    <L 1838>
                                // size += 1                                                      <L 1839>
                                var_388 = wp::add(var_364, var_387);
                            }
                            if (!var_385) {
                                // elif i == 4:                                                   <L 1840>
                                var_390 = (var_365 == var_389);
                                if (var_390) {
                                    // pos = True                                                 <L 1841>
                                    // size += 3                                                  <L 1842>
                                    var_393 = wp::add(var_364, var_392);
                                }
                                if (!var_390) {
                                    // elif i == 5:                                               <L 1843>
                                    var_395 = (var_365 == var_394);
                                    if (var_395) {
                                        // normal = True                                          <L 1844>
                                        // size += 3                                              <L 1845>
                                        var_398 = wp::add(var_364, var_397);
                                    }
                                    if (!var_395) {
                                        // elif i == 6:                                           <L 1846>
                                        var_400 = (var_365 == var_399);
                                        if (var_400) {
                                            // tangent = True                                     <L 1847>
                                            // size += 3                                          <L 1848>
                                            var_403 = wp::add(var_364, var_402);
                                        }
                                        var_404 = wp::where(var_400, var_401, var_363);
                                        var_405 = wp::where(var_400, var_403, var_364);
                                    }
                                    var_406 = wp::where(var_395, var_396, var_362);
                                    var_407 = wp::where(var_395, var_363, var_404);
                                    var_408 = wp::where(var_395, var_398, var_405);
                                }
                                var_409 = wp::where(var_390, var_391, var_361);
                                var_410 = wp::where(var_390, var_362, var_406);
                                var_411 = wp::where(var_390, var_363, var_407);
                                var_412 = wp::where(var_390, var_393, var_408);
                            }
                            var_413 = wp::where(var_385, var_386, var_360);
                            var_414 = wp::where(var_385, var_361, var_409);
                            var_415 = wp::where(var_385, var_362, var_410);
                            var_416 = wp::where(var_385, var_363, var_411);
                            var_417 = wp::where(var_385, var_388, var_412);
                        }
                        var_418 = wp::where(var_380, var_381, var_359);
                        var_419 = wp::where(var_380, var_360, var_413);
                        var_420 = wp::where(var_380, var_361, var_414);
                        var_421 = wp::where(var_380, var_362, var_415);
                        var_422 = wp::where(var_380, var_363, var_416);
                        var_423 = wp::where(var_380, var_383, var_417);
                    }
                    var_424 = wp::where(var_375, var_376, var_358);
                    var_425 = wp::where(var_375, var_359, var_418);
                    var_426 = wp::where(var_375, var_360, var_419);
                    var_427 = wp::where(var_375, var_361, var_420);
                    var_428 = wp::where(var_375, var_362, var_421);
                    var_429 = wp::where(var_375, var_363, var_422);
                    var_430 = wp::where(var_375, var_378, var_423);
                }
                var_431 = wp::where(var_370, var_371, var_357);
                var_432 = wp::where(var_370, var_358, var_424);
                var_433 = wp::where(var_370, var_359, var_425);
                var_434 = wp::where(var_370, var_360, var_426);
                var_435 = wp::where(var_370, var_361, var_427);
                var_436 = wp::where(var_370, var_362, var_428);
                var_437 = wp::where(var_370, var_363, var_429);
                var_438 = wp::where(var_370, var_373, var_430);
            }
            var_439 = wp::where(var_368, var_431, var_357);
            var_440 = wp::where(var_368, var_432, var_358);
            var_441 = wp::where(var_368, var_433, var_359);
            var_442 = wp::where(var_368, var_434, var_360);
            var_443 = wp::where(var_368, var_435, var_361);
            var_444 = wp::where(var_368, var_436, var_362);
            var_445 = wp::where(var_368, var_437, var_363);
            var_446 = wp::where(var_368, var_438, var_364);
            // if dataspec & (1 << i):                                                            <L 1827>
            var_449 = wp::lshift(var_448, var_447);
            var_450 = wp::bit_and(var_16, var_449);
            if (var_450) {
                // if i == 0:                                                                     <L 1828>
                var_452 = (var_447 == var_451);
                if (var_452) {
                    // found = True                                                               <L 1829>
                    // size += 1                                                                  <L 1830>
                    var_455 = wp::add(var_446, var_454);
                }
                if (!var_452) {
                    // elif i == 1:                                                               <L 1831>
                    var_457 = (var_447 == var_456);
                    if (var_457) {
                        // force = True                                                           <L 1832>
                        // size += 3                                                              <L 1833>
                        var_460 = wp::add(var_446, var_459);
                    }
                    if (!var_457) {
                        // elif i == 2:                                                           <L 1834>
                        var_462 = (var_447 == var_461);
                        if (var_462) {
                            // torque = True                                                      <L 1835>
                            // size += 3                                                          <L 1836>
                            var_465 = wp::add(var_446, var_464);
                        }
                        if (!var_462) {
                            // elif i == 3:                                                       <L 1837>
                            var_467 = (var_447 == var_466);
                            if (var_467) {
                                // dist = True                                                    <L 1838>
                                // size += 1                                                      <L 1839>
                                var_470 = wp::add(var_446, var_469);
                            }
                            if (!var_467) {
                                // elif i == 4:                                                   <L 1840>
                                var_472 = (var_447 == var_471);
                                if (var_472) {
                                    // pos = True                                                 <L 1841>
                                    // size += 3                                                  <L 1842>
                                    var_475 = wp::add(var_446, var_474);
                                }
                                if (!var_472) {
                                    // elif i == 5:                                               <L 1843>
                                    var_477 = (var_447 == var_476);
                                    if (var_477) {
                                        // normal = True                                          <L 1844>
                                        // size += 3                                              <L 1845>
                                        var_480 = wp::add(var_446, var_479);
                                    }
                                    if (!var_477) {
                                        // elif i == 6:                                           <L 1846>
                                        var_482 = (var_447 == var_481);
                                        if (var_482) {
                                            // tangent = True                                     <L 1847>
                                            // size += 3                                          <L 1848>
                                            var_485 = wp::add(var_446, var_484);
                                        }
                                        var_486 = wp::where(var_482, var_483, var_445);
                                        var_487 = wp::where(var_482, var_485, var_446);
                                    }
                                    var_488 = wp::where(var_477, var_478, var_444);
                                    var_489 = wp::where(var_477, var_445, var_486);
                                    var_490 = wp::where(var_477, var_480, var_487);
                                }
                                var_491 = wp::where(var_472, var_473, var_443);
                                var_492 = wp::where(var_472, var_444, var_488);
                                var_493 = wp::where(var_472, var_445, var_489);
                                var_494 = wp::where(var_472, var_475, var_490);
                            }
                            var_495 = wp::where(var_467, var_468, var_442);
                            var_496 = wp::where(var_467, var_443, var_491);
                            var_497 = wp::where(var_467, var_444, var_492);
                            var_498 = wp::where(var_467, var_445, var_493);
                            var_499 = wp::where(var_467, var_470, var_494);
                        }
                        var_500 = wp::where(var_462, var_463, var_441);
                        var_501 = wp::where(var_462, var_442, var_495);
                        var_502 = wp::where(var_462, var_443, var_496);
                        var_503 = wp::where(var_462, var_444, var_497);
                        var_504 = wp::where(var_462, var_445, var_498);
                        var_505 = wp::where(var_462, var_465, var_499);
                    }
                    var_506 = wp::where(var_457, var_458, var_440);
                    var_507 = wp::where(var_457, var_441, var_500);
                    var_508 = wp::where(var_457, var_442, var_501);
                    var_509 = wp::where(var_457, var_443, var_502);
                    var_510 = wp::where(var_457, var_444, var_503);
                    var_511 = wp::where(var_457, var_445, var_504);
                    var_512 = wp::where(var_457, var_460, var_505);
                }
                var_513 = wp::where(var_452, var_453, var_439);
                var_514 = wp::where(var_452, var_440, var_506);
                var_515 = wp::where(var_452, var_441, var_507);
                var_516 = wp::where(var_452, var_442, var_508);
                var_517 = wp::where(var_452, var_443, var_509);
                var_518 = wp::where(var_452, var_444, var_510);
                var_519 = wp::where(var_452, var_445, var_511);
                var_520 = wp::where(var_452, var_455, var_512);
            }
            var_521 = wp::where(var_450, var_513, var_439);
            var_522 = wp::where(var_450, var_514, var_440);
            var_523 = wp::where(var_450, var_515, var_441);
            var_524 = wp::where(var_450, var_516, var_442);
            var_525 = wp::where(var_450, var_517, var_443);
            var_526 = wp::where(var_450, var_518, var_444);
            var_527 = wp::where(var_450, var_519, var_445);
            var_528 = wp::where(var_450, var_520, var_446);
            // if dataspec & (1 << i):                                                            <L 1827>
            var_531 = wp::lshift(var_530, var_529);
            var_532 = wp::bit_and(var_16, var_531);
            if (var_532) {
                // if i == 0:                                                                     <L 1828>
                var_534 = (var_529 == var_533);
                if (var_534) {
                    // found = True                                                               <L 1829>
                    // size += 1                                                                  <L 1830>
                    var_537 = wp::add(var_528, var_536);
                }
                if (!var_534) {
                    // elif i == 1:                                                               <L 1831>
                    var_539 = (var_529 == var_538);
                    if (var_539) {
                        // force = True                                                           <L 1832>
                        // size += 3                                                              <L 1833>
                        var_542 = wp::add(var_528, var_541);
                    }
                    if (!var_539) {
                        // elif i == 2:                                                           <L 1834>
                        var_544 = (var_529 == var_543);
                        if (var_544) {
                            // torque = True                                                      <L 1835>
                            // size += 3                                                          <L 1836>
                            var_547 = wp::add(var_528, var_546);
                        }
                        if (!var_544) {
                            // elif i == 3:                                                       <L 1837>
                            var_549 = (var_529 == var_548);
                            if (var_549) {
                                // dist = True                                                    <L 1838>
                                // size += 1                                                      <L 1839>
                                var_552 = wp::add(var_528, var_551);
                            }
                            if (!var_549) {
                                // elif i == 4:                                                   <L 1840>
                                var_554 = (var_529 == var_553);
                                if (var_554) {
                                    // pos = True                                                 <L 1841>
                                    // size += 3                                                  <L 1842>
                                    var_557 = wp::add(var_528, var_556);
                                }
                                if (!var_554) {
                                    // elif i == 5:                                               <L 1843>
                                    var_559 = (var_529 == var_558);
                                    if (var_559) {
                                        // normal = True                                          <L 1844>
                                        // size += 3                                              <L 1845>
                                        var_562 = wp::add(var_528, var_561);
                                    }
                                    if (!var_559) {
                                        // elif i == 6:                                           <L 1846>
                                        var_564 = (var_529 == var_563);
                                        if (var_564) {
                                            // tangent = True                                     <L 1847>
                                            // size += 3                                          <L 1848>
                                            var_567 = wp::add(var_528, var_566);
                                        }
                                        var_568 = wp::where(var_564, var_565, var_527);
                                        var_569 = wp::where(var_564, var_567, var_528);
                                    }
                                    var_570 = wp::where(var_559, var_560, var_526);
                                    var_571 = wp::where(var_559, var_527, var_568);
                                    var_572 = wp::where(var_559, var_562, var_569);
                                }
                                var_573 = wp::where(var_554, var_555, var_525);
                                var_574 = wp::where(var_554, var_526, var_570);
                                var_575 = wp::where(var_554, var_527, var_571);
                                var_576 = wp::where(var_554, var_557, var_572);
                            }
                            var_577 = wp::where(var_549, var_550, var_524);
                            var_578 = wp::where(var_549, var_525, var_573);
                            var_579 = wp::where(var_549, var_526, var_574);
                            var_580 = wp::where(var_549, var_527, var_575);
                            var_581 = wp::where(var_549, var_552, var_576);
                        }
                        var_582 = wp::where(var_544, var_545, var_523);
                        var_583 = wp::where(var_544, var_524, var_577);
                        var_584 = wp::where(var_544, var_525, var_578);
                        var_585 = wp::where(var_544, var_526, var_579);
                        var_586 = wp::where(var_544, var_527, var_580);
                        var_587 = wp::where(var_544, var_547, var_581);
                    }
                    var_588 = wp::where(var_539, var_540, var_522);
                    var_589 = wp::where(var_539, var_523, var_582);
                    var_590 = wp::where(var_539, var_524, var_583);
                    var_591 = wp::where(var_539, var_525, var_584);
                    var_592 = wp::where(var_539, var_526, var_585);
                    var_593 = wp::where(var_539, var_527, var_586);
                    var_594 = wp::where(var_539, var_542, var_587);
                }
                var_595 = wp::where(var_534, var_535, var_521);
                var_596 = wp::where(var_534, var_522, var_588);
                var_597 = wp::where(var_534, var_523, var_589);
                var_598 = wp::where(var_534, var_524, var_590);
                var_599 = wp::where(var_534, var_525, var_591);
                var_600 = wp::where(var_534, var_526, var_592);
                var_601 = wp::where(var_534, var_527, var_593);
                var_602 = wp::where(var_534, var_537, var_594);
            }
            var_603 = wp::where(var_532, var_595, var_521);
            var_604 = wp::where(var_532, var_596, var_522);
            var_605 = wp::where(var_532, var_597, var_523);
            var_606 = wp::where(var_532, var_598, var_524);
            var_607 = wp::where(var_532, var_599, var_525);
            var_608 = wp::where(var_532, var_600, var_526);
            var_609 = wp::where(var_532, var_601, var_527);
            var_610 = wp::where(var_532, var_602, var_528);
            // num = dim // size  # number of slots                                               <L 1850>
            var_611 = wp::floordiv(var_19, var_610);
            // adr = sensor_adr[sensorid]                                                         <L 1852>
            var_612 = wp::address(var_sensor_adr, var_3);
            var_614 = wp::load(var_612);
            var_613 = wp::copy(var_614);
            // contactsensorid = sensor_adr_to_contact_adr[sensorid]                              <L 1853>
            var_615 = wp::address(var_sensor_adr_to_contact_adr, var_3);
            var_617 = wp::load(var_615);
            var_616 = wp::copy(var_617);
            // nmatch = sensor_contact_nmatch_in[worldid, contactsensorid]                        <L 1854>
            var_618 = wp::address(var_sensor_contact_nmatch_in, var_0, var_616);
            var_620 = wp::load(var_618);
            var_619 = wp::copy(var_620);
            // if reduce == 3:  # netforce                                                        <L 1856>
            var_622 = (var_26 == var_621);
            if (var_622) {
                // net_pos = wp.vec3(0.0)                                                         <L 1859>
                var_624 = wp::vec_t<3, wp::float32>(var_623);
                // net_force = wp.vec3(0.0)                                                       <L 1860>
                var_626 = wp::vec_t<3, wp::float32>(var_625);
                // net_torque = wp.vec3(0.0)                                                      <L 1861>
                var_628 = wp::vec_t<3, wp::float32>(var_627);
                // total_force_magnitude = float(0.0)                                             <L 1862>
                var_630 = wp::float(var_629);
                // for i in range(nmatch):                                                        <L 1864>
                var_631 = wp::range(var_619);
                start_for_0:;
                    if (iter_cmp(var_631) == 0) goto end_for_0;
                    var_632 = wp::iter_next(var_631);
                    // cid = sensor_contact_matchid_in[worldid, contactsensorid, i]               <L 1865>
                    var_633 = wp::address(var_sensor_contact_matchid_in, var_0, var_616, var_632);
                    var_635 = wp::load(var_633);
                    var_634 = wp::copy(var_635);
                    // dir = sensor_contact_direction_in[worldid, contactsensorid, i]             <L 1866>
                    var_636 = wp::address(var_sensor_contact_direction_in, var_0, var_616, var_632);
                    var_638 = wp::load(var_636);
                    var_637 = wp::copy(var_638);
                    // contact_forcetorque = support.contact_force_fn(                            <L 1868>
                    // opt_cone,                                                                  <L 1869>
                    // contact_frame_in,                                                          <L 1870>
                    // contact_friction_in,                                                       <L 1871>
                    // contact_dim_in,                                                            <L 1872>
                    // contact_efc_address_in,                                                    <L 1873>
                    // efc_force_in,                                                              <L 1874>
                    // njmax_in,                                                                  <L 1875>
                    // nacon_in,                                                                  <L 1876>
                    // worldid,                                                                   <L 1877>
                    // cid,                                                                       <L 1878>
                    // False,                                                                     <L 1879>
                    var_640 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_0, var_634, var_639);
                    // weight = wp.norm_l2(wp.spatial_top(contact_forcetorque))                   <L 1883>
                    var_641 = wp::spatial_top(var_640);
                    var_642 = norm_l2_0(var_641);
                    // contact_pos = contact_pos_in[cid]                                          <L 1884>
                    var_643 = wp::address(var_contact_pos_in, var_634);
                    var_645 = wp::load(var_643);
                    var_644 = wp::copy(var_645);
                    // net_pos += weight * contact_pos                                            <L 1885>
                    var_646 = wp::mul(var_642, var_644);
                    var_647 = wp::add(var_624, var_646);
                    // total_force_magnitude += weight                                            <L 1886>
                    var_648 = wp::add(var_630, var_642);
                    // contact_forcetorque *= dir                                                 <L 1889>
                    var_649 = wp::mul(var_640, var_637);
                    // force_local = wp.spatial_top(contact_forcetorque)                          <L 1890>
                    var_650 = wp::spatial_top(var_649);
                    // torque_local = wp.spatial_bottom(contact_forcetorque)                      <L 1891>
                    var_651 = wp::spatial_bottom(var_649);
                    // frame = contact_frame_in[cid]                                              <L 1893>
                    var_652 = wp::address(var_contact_frame_in, var_634);
                    var_654 = wp::load(var_652);
                    var_653 = wp::copy(var_654);
                    // frameT = wp.transpose(frame)                                               <L 1894>
                    var_655 = wp::transpose(var_653);
                    // force_global = frameT @ force_local                                        <L 1896>
                    var_656 = wp::mul(var_655, var_650);
                    // torque_global = frameT @ torque_local                                      <L 1897>
                    var_657 = wp::mul(var_655, var_651);
                    // net_force += force_global                                                  <L 1900>
                    var_658 = wp::add(var_626, var_656);
                    // net_torque += torque_global                                                <L 1901>
                    var_659 = wp::add(var_628, var_657);
                    // net_torque += wp.cross(contact_pos, force_global)                          <L 1903>
                    var_660 = wp::cross(var_644, var_656);
                    var_661 = wp::add(var_659, var_660);
                    wp::assign(var_624, var_647);
                    wp::assign(var_626, var_658);
                    wp::assign(var_628, var_661);
                    wp::assign(var_630, var_648);
                    goto start_for_0;
                end_for_0:;
                // net_pos /= wp.max(total_force_magnitude, MJ_MINVAL)                            <L 1906>
                var_663 = wp::max(var_630, var_662);
                var_664 = wp::div(var_624, var_663);
                // net_torque -= wp.cross(net_pos, net_force)                                     <L 1910>
                var_665 = wp::cross(var_664, var_626);
                var_666 = wp::sub(var_628, var_665);
                // adr_slot = adr                                                                 <L 1912>
                var_667 = wp::copy(var_613);
                // if found:                                                                      <L 1914>
                if (var_603) {
                    // out[adr_slot] = float(nmatch)                                              <L 1915>
                    var_668 = wp::float(var_619);
                    wp::array_store(var_11, var_667, var_668);
                    // adr_slot += 1                                                              <L 1916>
                    var_670 = wp::add(var_667, var_669);
                }
                var_671 = wp::where(var_603, var_670, var_667);
                // if force:                                                                      <L 1917>
                if (var_604) {
                    // out[adr_slot + 0] = net_force[0]                                           <L 1918>
                    var_673 = wp::extract(var_626, var_672);
                    var_675 = wp::add(var_671, var_674);
                    wp::array_store(var_11, var_675, var_673);
                    // out[adr_slot + 1] = net_force[1]                                           <L 1919>
                    var_677 = wp::extract(var_626, var_676);
                    var_679 = wp::add(var_671, var_678);
                    wp::array_store(var_11, var_679, var_677);
                    // out[adr_slot + 2] = net_force[2]                                           <L 1920>
                    var_681 = wp::extract(var_626, var_680);
                    var_683 = wp::add(var_671, var_682);
                    wp::array_store(var_11, var_683, var_681);
                    // adr_slot += 3                                                              <L 1921>
                    var_685 = wp::add(var_671, var_684);
                }
                var_686 = wp::where(var_604, var_685, var_671);
                // if torque:                                                                     <L 1922>
                if (var_605) {
                    // out[adr_slot + 0] = net_torque[0]                                          <L 1923>
                    var_688 = wp::extract(var_666, var_687);
                    var_690 = wp::add(var_686, var_689);
                    wp::array_store(var_11, var_690, var_688);
                    // out[adr_slot + 1] = net_torque[1]                                          <L 1924>
                    var_692 = wp::extract(var_666, var_691);
                    var_694 = wp::add(var_686, var_693);
                    wp::array_store(var_11, var_694, var_692);
                    // out[adr_slot + 2] = net_torque[2]                                          <L 1925>
                    var_696 = wp::extract(var_666, var_695);
                    var_698 = wp::add(var_686, var_697);
                    wp::array_store(var_11, var_698, var_696);
                    // adr_slot += 3                                                              <L 1926>
                    var_700 = wp::add(var_686, var_699);
                }
                var_701 = wp::where(var_605, var_700, var_686);
                // if dist:                                                                       <L 1927>
                if (var_606) {
                    // out[adr_slot] = 0.0                                                        <L 1928>
                    wp::array_store(var_11, var_701, var_702);
                    // adr_slot += 1                                                              <L 1929>
                    var_704 = wp::add(var_701, var_703);
                }
                var_705 = wp::where(var_606, var_704, var_701);
                // if pos:                                                                        <L 1930>
                if (var_607) {
                    // out[adr_slot + 0] = net_pos[0]                                             <L 1931>
                    var_707 = wp::extract(var_664, var_706);
                    var_709 = wp::add(var_705, var_708);
                    wp::array_store(var_11, var_709, var_707);
                    // out[adr_slot + 1] = net_pos[1]                                             <L 1932>
                    var_711 = wp::extract(var_664, var_710);
                    var_713 = wp::add(var_705, var_712);
                    wp::array_store(var_11, var_713, var_711);
                    // out[adr_slot + 2] = net_pos[2]                                             <L 1933>
                    var_715 = wp::extract(var_664, var_714);
                    var_717 = wp::add(var_705, var_716);
                    wp::array_store(var_11, var_717, var_715);
                    // adr_slot += 3                                                              <L 1934>
                    var_719 = wp::add(var_705, var_718);
                }
                var_720 = wp::where(var_607, var_719, var_705);
                // if normal:                                                                     <L 1935>
                if (var_608) {
                    // out[adr_slot + 0] = 1.0                                                    <L 1936>
                    var_723 = wp::add(var_720, var_722);
                    wp::array_store(var_11, var_723, var_721);
                    // out[adr_slot + 1] = 0.0                                                    <L 1937>
                    var_726 = wp::add(var_720, var_725);
                    wp::array_store(var_11, var_726, var_724);
                    // out[adr_slot + 2] = 0.0                                                    <L 1938>
                    var_729 = wp::add(var_720, var_728);
                    wp::array_store(var_11, var_729, var_727);
                    // adr_slot += 3                                                              <L 1939>
                    var_731 = wp::add(var_720, var_730);
                }
                var_732 = wp::where(var_608, var_731, var_720);
                // if tangent:                                                                    <L 1940>
                if (var_609) {
                    // out[adr_slot + 0] = 0.0                                                    <L 1941>
                    var_735 = wp::add(var_732, var_734);
                    wp::array_store(var_11, var_735, var_733);
                    // out[adr_slot + 1] = 1.0                                                    <L 1942>
                    var_738 = wp::add(var_732, var_737);
                    wp::array_store(var_11, var_738, var_736);
                    // out[adr_slot + 2] = 0.0                                                    <L 1943>
                    var_741 = wp::add(var_732, var_740);
                    wp::array_store(var_11, var_741, var_739);
                }
            }
            if (!var_622) {
                // nslots = wp.min(nmatch, num)                                                   <L 1945>
                var_742 = wp::min(var_619, var_611);
                // for i in range(nslots):                                                        <L 1946>
                var_743 = wp::range(var_742);
                start_for_2:;
                    if (iter_cmp(var_743) == 0) goto end_for_2;
                    var_744 = wp::iter_next(var_743);
                    // cid = sensor_contact_matchid_in[worldid, contactsensorid, i]               <L 1948>
                    var_745 = wp::address(var_sensor_contact_matchid_in, var_0, var_616, var_744);
                    var_747 = wp::load(var_745);
                    var_746 = wp::copy(var_747);
                    // dir = sensor_contact_direction_in[worldid, contactsensorid, i]             <L 1951>
                    var_748 = wp::address(var_sensor_contact_direction_in, var_0, var_616, var_744);
                    var_750 = wp::load(var_748);
                    var_749 = wp::copy(var_750);
                    // adr_slot = adr + i * size                                                  <L 1953>
                    var_751 = wp::mul(var_744, var_610);
                    var_752 = wp::add(var_613, var_751);
                    // if found:                                                                  <L 1955>
                    if (var_603) {
                        // out[adr_slot] = float(nmatch)                                          <L 1956>
                        var_753 = wp::float(var_619);
                        wp::array_store(var_11, var_752, var_753);
                        // adr_slot += 1                                                          <L 1957>
                        var_755 = wp::add(var_752, var_754);
                    }
                    var_756 = wp::where(var_603, var_755, var_752);
                    // if force or torque:                                                        <L 1958>
                    var_757 = var_604;
                    if (!var_757) {
                        var_757 = var_757 || var_605;
                    }
                    if (var_757) {
                        // contact_forcetorque = support.contact_force_fn(                        <L 1959>
                        // opt_cone,                                                              <L 1960>
                        // contact_frame_in,                                                      <L 1961>
                        // contact_friction_in,                                                   <L 1962>
                        // contact_dim_in,                                                        <L 1963>
                        // contact_efc_address_in,                                                <L 1964>
                        // efc_force_in,                                                          <L 1965>
                        // njmax_in,                                                              <L 1966>
                        // nacon_in,                                                              <L 1967>
                        // worldid,                                                               <L 1968>
                        // cid,                                                                   <L 1969>
                        // False,                                                                 <L 1970>
                        var_759 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_0, var_746, var_758);
                    }
                    // if force:                                                                  <L 1972>
                    if (var_604) {
                        // out[adr_slot + 0] = contact_forcetorque[0]                             <L 1973>
                        var_761 = wp::extract(var_759, var_760);
                        var_763 = wp::add(var_756, var_762);
                        wp::array_store(var_11, var_763, var_761);
                        // out[adr_slot + 1] = contact_forcetorque[1]                             <L 1974>
                        var_765 = wp::extract(var_759, var_764);
                        var_767 = wp::add(var_756, var_766);
                        wp::array_store(var_11, var_767, var_765);
                        // out[adr_slot + 2] = dir * contact_forcetorque[2]                       <L 1975>
                        var_769 = wp::extract(var_759, var_768);
                        var_770 = wp::mul(var_749, var_769);
                        var_772 = wp::add(var_756, var_771);
                        wp::array_store(var_11, var_772, var_770);
                        // adr_slot += 3                                                          <L 1976>
                        var_774 = wp::add(var_756, var_773);
                    }
                    var_775 = wp::where(var_604, var_774, var_756);
                    // if torque:                                                                 <L 1977>
                    if (var_605) {
                        // out[adr_slot + 0] = contact_forcetorque[3]                             <L 1978>
                        var_777 = wp::extract(var_759, var_776);
                        var_779 = wp::add(var_775, var_778);
                        wp::array_store(var_11, var_779, var_777);
                        // out[adr_slot + 1] = contact_forcetorque[4]                             <L 1979>
                        var_781 = wp::extract(var_759, var_780);
                        var_783 = wp::add(var_775, var_782);
                        wp::array_store(var_11, var_783, var_781);
                        // out[adr_slot + 2] = dir * contact_forcetorque[5]                       <L 1980>
                        var_785 = wp::extract(var_759, var_784);
                        var_786 = wp::mul(var_749, var_785);
                        var_788 = wp::add(var_775, var_787);
                        wp::array_store(var_11, var_788, var_786);
                        // adr_slot += 3                                                          <L 1981>
                        var_790 = wp::add(var_775, var_789);
                    }
                    var_791 = wp::where(var_605, var_790, var_775);
                    // if dist:                                                                   <L 1982>
                    if (var_606) {
                        // out[adr_slot] = contact_dist_in[cid]                                   <L 1983>
                        var_792 = wp::address(var_contact_dist_in, var_746);
                        var_793 = wp::load(var_792);
                        wp::array_store(var_11, var_791, var_793);
                        // adr_slot += 1                                                          <L 1984>
                        var_795 = wp::add(var_791, var_794);
                    }
                    var_796 = wp::where(var_606, var_795, var_791);
                    // if pos:                                                                    <L 1985>
                    if (var_607) {
                        // contact_pos = contact_pos_in[cid]                                      <L 1986>
                        var_797 = wp::address(var_contact_pos_in, var_746);
                        var_799 = wp::load(var_797);
                        var_798 = wp::copy(var_799);
                        // out[adr_slot + 0] = contact_pos[0]                                     <L 1987>
                        var_801 = wp::extract(var_798, var_800);
                        var_803 = wp::add(var_796, var_802);
                        wp::array_store(var_11, var_803, var_801);
                        // out[adr_slot + 1] = contact_pos[1]                                     <L 1988>
                        var_805 = wp::extract(var_798, var_804);
                        var_807 = wp::add(var_796, var_806);
                        wp::array_store(var_11, var_807, var_805);
                        // out[adr_slot + 2] = contact_pos[2]                                     <L 1989>
                        var_809 = wp::extract(var_798, var_808);
                        var_811 = wp::add(var_796, var_810);
                        wp::array_store(var_11, var_811, var_809);
                        // adr_slot += 3                                                          <L 1990>
                        var_813 = wp::add(var_796, var_812);
                    }
                    var_814 = wp::where(var_607, var_813, var_796);
                    // if normal:                                                                 <L 1991>
                    if (var_608) {
                        // contact_normal = contact_frame_in[cid][0]                              <L 1992>
                        var_815 = wp::address(var_contact_frame_in, var_746);
                        var_818 = wp::load(var_815);
                        var_817 = wp::extract(var_818, var_816);
                        // out[adr_slot + 0] = dir * contact_normal[0]                            <L 1993>
                        var_820 = wp::extract(var_817, var_819);
                        var_821 = wp::mul(var_749, var_820);
                        var_823 = wp::add(var_814, var_822);
                        wp::array_store(var_11, var_823, var_821);
                        // out[adr_slot + 1] = dir * contact_normal[1]                            <L 1994>
                        var_825 = wp::extract(var_817, var_824);
                        var_826 = wp::mul(var_749, var_825);
                        var_828 = wp::add(var_814, var_827);
                        wp::array_store(var_11, var_828, var_826);
                        // out[adr_slot + 2] = dir * contact_normal[2]                            <L 1995>
                        var_830 = wp::extract(var_817, var_829);
                        var_831 = wp::mul(var_749, var_830);
                        var_833 = wp::add(var_814, var_832);
                        wp::array_store(var_11, var_833, var_831);
                        // adr_slot += 3                                                          <L 1996>
                        var_835 = wp::add(var_814, var_834);
                    }
                    var_836 = wp::where(var_608, var_835, var_814);
                    // if tangent:                                                                <L 1997>
                    if (var_609) {
                        // contact_tangent = contact_frame_in[cid][1]                             <L 1998>
                        var_837 = wp::address(var_contact_frame_in, var_746);
                        var_840 = wp::load(var_837);
                        var_839 = wp::extract(var_840, var_838);
                        // out[adr_slot + 0] = dir * contact_tangent[0]                           <L 1999>
                        var_842 = wp::extract(var_839, var_841);
                        var_843 = wp::mul(var_749, var_842);
                        var_845 = wp::add(var_836, var_844);
                        wp::array_store(var_11, var_845, var_843);
                        // out[adr_slot + 1] = dir * contact_tangent[1]                           <L 2000>
                        var_847 = wp::extract(var_839, var_846);
                        var_848 = wp::mul(var_749, var_847);
                        var_850 = wp::add(var_836, var_849);
                        wp::array_store(var_11, var_850, var_848);
                        // out[adr_slot + 2] = dir * contact_tangent[2]                           <L 2001>
                        var_852 = wp::extract(var_839, var_851);
                        var_853 = wp::mul(var_749, var_852);
                        var_855 = wp::add(var_836, var_854);
                        wp::array_store(var_11, var_855, var_853);
                    }
                    goto start_for_2;
                end_for_2:;
                // for i in range(nmatch, num):                                                   <L 2004>
                var_856 = wp::range(var_619, var_611);
                start_for_4:;
                    if (iter_cmp(var_856) == 0) goto end_for_4;
                    var_857 = wp::iter_next(var_856);
                    // for j in range(size):                                                      <L 2005>
                    var_858 = wp::range(var_610);
                    start_for_6:;
                        if (iter_cmp(var_858) == 0) goto end_for_6;
                        var_859 = wp::iter_next(var_858);
                        // out[adr + i * size + j] = 0.0                                          <L 2006>
                        var_861 = wp::mul(var_857, var_610);
                        var_862 = wp::add(var_613, var_861);
                        var_863 = wp::add(var_862, var_859);
                        wp::array_store(var_11, var_863, var_860);
                        goto start_for_6;
                    end_for_6:;
                    goto start_for_4;
                end_for_4:;
            }
            var_864 = wp::where(var_622, var_632, var_857);
            var_865 = wp::where(var_622, var_634, var_746);
            var_866 = wp::where(var_622, var_637, var_749);
            var_867 = wp::where(var_622, var_649, var_759);
            var_868 = wp::where(var_622, var_644, var_798);
            var_869 = wp::where(var_622, var_732, var_836);
        }
        if (!var_13) {
            // elif sensortype == SensorType.ACCELEROMETER:                                       <L 2008>
            var_871 = (var_6 == var_870);
            if (var_871) {
                // vec3 = _accelerometer(                                                         <L 2009>
                // body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cvel_in, cacc_in, worldid, objid       <L 2010>
                var_872 = _accelerometer_0(var_body_rootid, var_site_bodyid, var_site_xpos_in, var_site_xmat_in, var_subtree_com_in, var_cvel_in, var_cacc_in, var_0, var_9);
                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 2012>
                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_873, var_872, var_11);
            }
            if (!var_871) {
                // elif sensortype == SensorType.FORCE:                                           <L 2013>
                var_875 = (var_6 == var_874);
                if (var_875) {
                    // vec3 = _force(site_bodyid, site_xmat_in, cfrc_int_in, worldid, objid)       <L 2014>
                    var_876 = _force_0(var_site_bodyid, var_site_xmat_in, var_cfrc_int_in, var_0, var_9);
                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 2015>
                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_877, var_876, var_11);
                }
                if (!var_875) {
                    // elif sensortype == SensorType.TORQUE:                                      <L 2016>
                    var_879 = (var_6 == var_878);
                    if (var_879) {
                        // vec3 = _torque(body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cfrc_int_in, worldid, objid)       <L 2017>
                        var_880 = _torque_0(var_body_rootid, var_site_bodyid, var_site_xpos_in, var_site_xmat_in, var_subtree_com_in, var_cfrc_int_in, var_0, var_9);
                        // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 2018>
                        _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_881, var_880, var_11);
                    }
                    if (!var_879) {
                        // elif sensortype == SensorType.ACTUATORFRC:                             <L 2019>
                        var_883 = (var_6 == var_882);
                        if (var_883) {
                            // val = _actuator_force(actuator_force_in, worldid, objid)           <L 2020>
                            var_884 = _actuator_force_0(var_actuator_force_in, var_0, var_9);
                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 2021>
                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_884, var_11);
                        }
                        if (!var_883) {
                            // elif sensortype == SensorType.JOINTACTFRC:                         <L 2022>
                            var_886 = (var_6 == var_885);
                            if (var_886) {
                                // val = _joint_actuator_force(jnt_dofadr, qfrc_actuator_in, worldid, objid)       <L 2023>
                                var_887 = _joint_actuator_force_0(var_jnt_dofadr, var_qfrc_actuator_in, var_0, var_9);
                                // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 2024>
                                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_887, var_11);
                            }
                            if (!var_886) {
                                // elif sensortype == SensorType.FRAMELINACC:                     <L 2025>
                                var_889 = (var_6 == var_888);
                                if (var_889) {
                                    // objtype = sensor_objtype[sensorid]                         <L 2026>
                                    var_890 = wp::address(var_sensor_objtype, var_3);
                                    var_892 = wp::load(var_890);
                                    var_891 = wp::copy(var_892);
                                    // vec3 = _framelinacc(                                       <L 2027>
                                    // body_rootid,                                               <L 2028>
                                    // geom_bodyid,                                               <L 2029>
                                    // site_bodyid,                                               <L 2030>
                                    // cam_bodyid,                                                <L 2031>
                                    // xpos_in,                                                   <L 2032>
                                    // xipos_in,                                                  <L 2033>
                                    // geom_xpos_in,                                              <L 2034>
                                    // site_xpos_in,                                              <L 2035>
                                    // cam_xpos_in,                                               <L 2036>
                                    // subtree_com_in,                                            <L 2037>
                                    // cvel_in,                                                   <L 2038>
                                    // cacc_in,                                                   <L 2039>
                                    // worldid,                                                   <L 2040>
                                    // objid,                                                     <L 2041>
                                    // objtype,                                                   <L 2042>
                                    var_893 = _framelinacc_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_cacc_in, var_0, var_9, var_891);
                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 2044>
                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_894, var_893, var_11);
                                }
                                if (!var_889) {
                                    // elif sensortype == SensorType.FRAMEANGACC:                 <L 2045>
                                    var_896 = (var_6 == var_895);
                                    if (var_896) {
                                        // objtype = sensor_objtype[sensorid]                     <L 2046>
                                        var_897 = wp::address(var_sensor_objtype, var_3);
                                        var_899 = wp::load(var_897);
                                        var_898 = wp::copy(var_899);
                                        // vec3 = _frameangacc(                                   <L 2047>
                                        // geom_bodyid,                                           <L 2048>
                                        // site_bodyid,                                           <L 2049>
                                        // cam_bodyid,                                            <L 2050>
                                        // cacc_in,                                               <L 2051>
                                        // worldid,                                               <L 2052>
                                        // objid,                                                 <L 2053>
                                        // objtype,                                               <L 2054>
                                        var_900 = _frameangacc_0(var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_cacc_in, var_0, var_9, var_898);
                                        // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 2056>
                                        _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_901, var_900, var_11);
                                    }
                                }
                                var_902 = wp::where(var_889, var_891, var_898);
                                var_903 = wp::where(var_889, var_893, var_900);
                            }
                        }
                        var_904 = wp::where(var_883, var_884, var_887);
                    }
                    var_905 = wp::where(var_879, var_880, var_903);
                }
                var_906 = wp::where(var_875, var_876, var_905);
            }
            var_907 = wp::where(var_871, var_872, var_906);
        }
        var_908 = wp::where(var_13, var_22, var_902);
}



extern "C" __global__ void _energy_pos_passive_joint_6f3b6468_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qpos_spring,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_jnt_stiffness,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_stiffnesspoly,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_out)
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
        bool var_18;
        const wp::float32 var_19 = 0.0;
        bool var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        bool var_24;
        const wp::int32 var_25 = 1;
        wp::float32 var_26;
        const wp::float32 var_27 = 0.0;
        bool var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::shape_t* var_35;
        const wp::int32 var_36 = 0;
        wp::int32 var_37;
        wp::shape_t var_38;
        wp::int32 var_39;
        const wp::int32 var_40 = 0;
        bool var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::float32* var_44;
        const wp::int32 var_45 = 0;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        wp::float32* var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        const wp::int32 var_60 = 2;
        wp::int32 var_61;
        wp::float32* var_62;
        const wp::int32 var_63 = 2;
        wp::int32 var_64;
        wp::float32* var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::vec_t<3, wp::float32> var_69;
        const wp::int32 var_70 = 3;
        wp::int32 var_71;
        wp::float32* var_72;
        const wp::int32 var_73 = 4;
        wp::int32 var_74;
        wp::float32* var_75;
        const wp::int32 var_76 = 5;
        wp::int32 var_77;
        wp::float32* var_78;
        const wp::int32 var_79 = 6;
        wp::int32 var_80;
        wp::float32* var_81;
        wp::quat_t<wp::float32> var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::quat_t<wp::float32> var_87;
        const wp::int32 var_88 = 3;
        wp::int32 var_89;
        wp::float32* var_90;
        const wp::int32 var_91 = 4;
        wp::int32 var_92;
        wp::float32* var_93;
        const wp::int32 var_94 = 5;
        wp::int32 var_95;
        wp::float32* var_96;
        const wp::int32 var_97 = 6;
        wp::int32 var_98;
        wp::float32* var_99;
        wp::quat_t<wp::float32> var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        const wp::int32 var_108 = 0;
        wp::float32 var_109;
        const wp::int32 var_110 = 0;
        wp::float32 var_111;
        wp::float32 var_112;
        const wp::float32 var_113 = 0.0;
        wp::vec_t<2, wp::float32> var_114;
        wp::vec_t<2, wp::float32> var_115;
        const wp::int32 var_116 = 1;
        bool var_117;
        const wp::int32 var_118 = 0;
        wp::int32 var_119;
        wp::float32* var_120;
        const wp::int32 var_121 = 1;
        wp::int32 var_122;
        wp::float32* var_123;
        const wp::int32 var_124 = 2;
        wp::int32 var_125;
        wp::float32* var_126;
        const wp::int32 var_127 = 3;
        wp::int32 var_128;
        wp::float32* var_129;
        wp::quat_t<wp::float32> var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::quat_t<wp::float32> var_135;
        const wp::int32 var_136 = 0;
        wp::int32 var_137;
        wp::float32* var_138;
        const wp::int32 var_139 = 1;
        wp::int32 var_140;
        wp::float32* var_141;
        const wp::int32 var_142 = 2;
        wp::int32 var_143;
        wp::float32* var_144;
        const wp::int32 var_145 = 3;
        wp::int32 var_146;
        wp::float32* var_147;
        wp::quat_t<wp::float32> var_148;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        wp::float32 var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 0;
        wp::float32 var_156;
        const wp::float32 var_157 = 0.0;
        wp::vec_t<2, wp::float32> var_158;
        wp::vec_t<2, wp::float32> var_159;
        bool var_160;
        const wp::int32 var_161 = 2;
        bool var_162;
        const wp::int32 var_163 = 3;
        bool var_164;
        wp::float32* var_165;
        wp::float32* var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        const wp::int32 var_170 = 0;
        wp::float32 var_171;
        const wp::float32 var_172 = 0.0;
        wp::vec_t<2, wp::float32> var_173;
        wp::vec_t<2, wp::float32> var_174;
        wp::vec_t<2, wp::float32> var_175;
        wp::quat_t<wp::float32> var_176;
        wp::vec_t<2, wp::float32> var_177;
        //---------
        // forward
        // def _energy_pos_passive_joint(                                                         <L 2797>
        // worldid, jntid = wp.tid()                                                              <L 2809>
        builtin_tid2d(var_0, var_1);
        // jnt_stiffness_id = worldid % jnt_stiffness.shape[0]                                    <L 2810>
        var_2 = &(var_jnt_stiffness.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // stiffness = jnt_stiffness[jnt_stiffness_id, jntid]                                     <L 2811>
        var_7 = wp::address(var_jnt_stiffness, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], jntid]                 <L 2812>
        var_10 = &(var_jnt_stiffnesspoly.shape);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::mod(var_0, var_12);
        var_15 = wp::address(var_jnt_stiffnesspoly, var_14, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // if stiffness == 0.0 and spoly[0] == 0.0 and spoly[1] == 0.0:                           <L 2814>
        var_20 = (var_8 == var_19);
        var_18 = var_20;
        if (var_18) {
            var_22 = wp::extract(var_16, var_21);
            var_24 = (var_22 == var_23);
            var_18 = var_18 && var_24;
        }
        if (var_18) {
            var_26 = wp::extract(var_16, var_25);
            var_28 = (var_26 == var_27);
            var_18 = var_18 && var_28;
        }
        if (var_18) {
            // return                                                                             <L 2815>
            return;
        }
        // padr = jnt_qposadr[jntid]                                                              <L 2817>
        var_29 = wp::address(var_jnt_qposadr, var_1);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // jnttype = jnt_type[jntid]                                                              <L 2818>
        var_32 = wp::address(var_jnt_type, var_1);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // qpos_spring_id = worldid % qpos_spring.shape[0]                                        <L 2819>
        var_35 = &(var_qpos_spring.shape);
        var_38 = wp::load(var_35);
        var_37 = wp::extract(var_38, var_36);
        var_39 = wp::mod(var_0, var_37);
        // if jnttype == JointType.FREE:                                                          <L 2821>
        var_41 = (var_33 == var_40);
        if (var_41) {
            // dif0 = wp.vec3(                                                                    <L 2822>
            // qpos_in[worldid, padr + 0] - qpos_spring[qpos_spring_id, padr + 0],                <L 2823>
            var_43 = wp::add(var_30, var_42);
            var_44 = wp::address(var_qpos_in, var_0, var_43);
            var_46 = wp::add(var_30, var_45);
            var_47 = wp::address(var_qpos_spring, var_39, var_46);
            var_49 = wp::load(var_44);
            var_50 = wp::load(var_47);
            var_48 = wp::sub(var_49, var_50);
            // qpos_in[worldid, padr + 1] - qpos_spring[qpos_spring_id, padr + 1],                <L 2824>
            var_52 = wp::add(var_30, var_51);
            var_53 = wp::address(var_qpos_in, var_0, var_52);
            var_55 = wp::add(var_30, var_54);
            var_56 = wp::address(var_qpos_spring, var_39, var_55);
            var_58 = wp::load(var_53);
            var_59 = wp::load(var_56);
            var_57 = wp::sub(var_58, var_59);
            // qpos_in[worldid, padr + 2] - qpos_spring[qpos_spring_id, padr + 2],                <L 2825>
            var_61 = wp::add(var_30, var_60);
            var_62 = wp::address(var_qpos_in, var_0, var_61);
            var_64 = wp::add(var_30, var_63);
            var_65 = wp::address(var_qpos_spring, var_39, var_64);
            var_67 = wp::load(var_62);
            var_68 = wp::load(var_65);
            var_66 = wp::sub(var_67, var_68);
            var_69 = wp::vec_t<3, wp::float32>(var_48, var_57, var_66);
            // quat1 = wp.quat(                                                                   <L 2829>
            // qpos_in[worldid, padr + 3],                                                        <L 2830>
            var_71 = wp::add(var_30, var_70);
            var_72 = wp::address(var_qpos_in, var_0, var_71);
            // qpos_in[worldid, padr + 4],                                                        <L 2831>
            var_74 = wp::add(var_30, var_73);
            var_75 = wp::address(var_qpos_in, var_0, var_74);
            // qpos_in[worldid, padr + 5],                                                        <L 2832>
            var_77 = wp::add(var_30, var_76);
            var_78 = wp::address(var_qpos_in, var_0, var_77);
            // qpos_in[worldid, padr + 6],                                                        <L 2833>
            var_80 = wp::add(var_30, var_79);
            var_81 = wp::address(var_qpos_in, var_0, var_80);
            var_83 = wp::load(var_72);
            var_84 = wp::load(var_75);
            var_85 = wp::load(var_78);
            var_86 = wp::load(var_81);
            var_82 = wp::quat_t<wp::float32>(var_83, var_84, var_85, var_86);
            // quat1 = wp.normalize(quat1)                                                        <L 2835>
            var_87 = wp::normalize(var_82);
            // quat_spring = wp.quat(                                                             <L 2837>
            // qpos_spring[qpos_spring_id, padr + 3],                                             <L 2838>
            var_89 = wp::add(var_30, var_88);
            var_90 = wp::address(var_qpos_spring, var_39, var_89);
            // qpos_spring[qpos_spring_id, padr + 4],                                             <L 2839>
            var_92 = wp::add(var_30, var_91);
            var_93 = wp::address(var_qpos_spring, var_39, var_92);
            // qpos_spring[qpos_spring_id, padr + 5],                                             <L 2840>
            var_95 = wp::add(var_30, var_94);
            var_96 = wp::address(var_qpos_spring, var_39, var_95);
            // qpos_spring[qpos_spring_id, padr + 6],                                             <L 2841>
            var_98 = wp::add(var_30, var_97);
            var_99 = wp::address(var_qpos_spring, var_39, var_98);
            var_101 = wp::load(var_90);
            var_102 = wp::load(var_93);
            var_103 = wp::load(var_96);
            var_104 = wp::load(var_99);
            var_100 = wp::quat_t<wp::float32>(var_101, var_102, var_103, var_104);
            // dif1 = math.quat_sub(quat1, quat_spring)                                           <L 2844>
            var_105 = quat_sub_0(var_87, var_100);
            // r0 = wp.length(dif0)                                                               <L 2846>
            var_106 = wp::length(var_69);
            // r1 = wp.length(dif1)                                                               <L 2847>
            var_107 = wp::length(var_105);
            // energy = wp.vec2(                                                                  <L 2849>
            // poly_potential(stiffness, spoly, r0, 0) + poly_potential(stiffness, spoly, r1, 0),       <L 2850>
            var_109 = poly_potential_0(var_8, var_16, var_106, var_108);
            var_111 = poly_potential_0(var_8, var_16, var_107, var_110);
            var_112 = wp::add(var_109, var_111);
            // 0.0,                                                                               <L 2851>
            var_114 = wp::vec_t<2, wp::float32>(var_112, var_113);
            // wp.atomic_add(energy_out, worldid, energy)                                         <L 2854>
            var_115 = wp::atomic_add(var_energy_out, var_0, var_114);
        }
        if (!var_41) {
            // elif jnttype == JointType.BALL:                                                    <L 2856>
            var_117 = (var_33 == var_116);
            if (var_117) {
                // quat = wp.quat(                                                                <L 2857>
                // qpos_in[worldid, padr + 0],                                                    <L 2858>
                var_119 = wp::add(var_30, var_118);
                var_120 = wp::address(var_qpos_in, var_0, var_119);
                // qpos_in[worldid, padr + 1],                                                    <L 2859>
                var_122 = wp::add(var_30, var_121);
                var_123 = wp::address(var_qpos_in, var_0, var_122);
                // qpos_in[worldid, padr + 2],                                                    <L 2860>
                var_125 = wp::add(var_30, var_124);
                var_126 = wp::address(var_qpos_in, var_0, var_125);
                // qpos_in[worldid, padr + 3],                                                    <L 2861>
                var_128 = wp::add(var_30, var_127);
                var_129 = wp::address(var_qpos_in, var_0, var_128);
                var_131 = wp::load(var_120);
                var_132 = wp::load(var_123);
                var_133 = wp::load(var_126);
                var_134 = wp::load(var_129);
                var_130 = wp::quat_t<wp::float32>(var_131, var_132, var_133, var_134);
                // quat = wp.normalize(quat)                                                      <L 2863>
                var_135 = wp::normalize(var_130);
                // quat_spring = wp.quat(                                                         <L 2865>
                // qpos_spring[qpos_spring_id, padr + 0],                                         <L 2866>
                var_137 = wp::add(var_30, var_136);
                var_138 = wp::address(var_qpos_spring, var_39, var_137);
                // qpos_spring[qpos_spring_id, padr + 1],                                         <L 2867>
                var_140 = wp::add(var_30, var_139);
                var_141 = wp::address(var_qpos_spring, var_39, var_140);
                // qpos_spring[qpos_spring_id, padr + 2],                                         <L 2868>
                var_143 = wp::add(var_30, var_142);
                var_144 = wp::address(var_qpos_spring, var_39, var_143);
                // qpos_spring[qpos_spring_id, padr + 3],                                         <L 2869>
                var_146 = wp::add(var_30, var_145);
                var_147 = wp::address(var_qpos_spring, var_39, var_146);
                var_149 = wp::load(var_138);
                var_150 = wp::load(var_141);
                var_151 = wp::load(var_144);
                var_152 = wp::load(var_147);
                var_148 = wp::quat_t<wp::float32>(var_149, var_150, var_151, var_152);
                // dif = math.quat_sub(quat, quat_spring)                                         <L 2872>
                var_153 = quat_sub_0(var_135, var_148);
                // r = wp.length(dif)                                                             <L 2873>
                var_154 = wp::length(var_153);
                // energy = wp.vec2(                                                              <L 2874>
                // poly_potential(stiffness, spoly, r, 0),                                        <L 2875>
                var_156 = poly_potential_0(var_8, var_16, var_154, var_155);
                // 0.0,                                                                           <L 2876>
                var_158 = wp::vec_t<2, wp::float32>(var_156, var_157);
                // wp.atomic_add(energy_out, worldid, energy)                                     <L 2878>
                var_159 = wp::atomic_add(var_energy_out, var_0, var_158);
            }
            if (!var_117) {
                // elif jnttype == JointType.SLIDE or jnttype == JointType.HINGE:                 <L 2879>
                var_162 = (var_33 == var_161);
                var_160 = var_162;
                if (!var_160) {
                    var_164 = (var_33 == var_163);
                    var_160 = var_160 || var_164;
                }
                if (var_160) {
                    // dif_ = qpos_in[worldid, padr] - qpos_spring[qpos_spring_id, padr]          <L 2880>
                    var_165 = wp::address(var_qpos_in, var_0, var_30);
                    var_166 = wp::address(var_qpos_spring, var_39, var_30);
                    var_168 = wp::load(var_165);
                    var_169 = wp::load(var_166);
                    var_167 = wp::sub(var_168, var_169);
                    // energy = wp.vec2(                                                          <L 2881>
                    // poly_potential(stiffness, spoly, dif_, 0),                                 <L 2882>
                    var_171 = poly_potential_0(var_8, var_16, var_167, var_170);
                    // 0.0,                                                                       <L 2883>
                    var_173 = wp::vec_t<2, wp::float32>(var_171, var_172);
                    // wp.atomic_add(energy_out, worldid, energy)                                 <L 2885>
                    var_174 = wp::atomic_add(var_energy_out, var_0, var_173);
                }
            }
            var_175 = wp::where(var_117, var_158, var_173);
        }
        var_176 = wp::where(var_41, var_100, var_148);
        var_177 = wp::where(var_41, var_114, var_175);
}



extern "C" __global__ void _tendon_actuator_force_a7ed58b7_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::int32> var_sensor_tendonactfrc_adr,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32* var_7;
        const wp::int32 var_8 = 3;
        bool var_9;
        wp::int32 var_10;
        wp::vec_t<2, wp::int32>* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::vec_t<2, wp::int32> var_14;
        wp::int32* var_15;
        bool var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        //---------
        // forward
        // def _tendon_actuator_force(                                                            <L 1598>
        // worldid, tenactfrcid, actid = wp.tid()                                                 <L 1610>
        builtin_tid3d(var_0, var_1, var_2);
        // sensorid = sensor_tendonactfrc_adr[tenactfrcid]                                        <L 1611>
        var_3 = wp::address(var_sensor_tendonactfrc_adr, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // if actuator_trntype[actid] == TrnType.TENDON and actuator_trnid[actid][0] == sensor_objid[sensorid]:       <L 1613>
        var_7 = wp::address(var_actuator_trntype, var_2);
        var_10 = wp::load(var_7);
        var_9 = (var_10 == var_8);
        var_6 = var_9;
        if (var_6) {
            var_11 = wp::address(var_actuator_trnid, var_2);
            var_14 = wp::load(var_11);
            var_13 = wp::extract(var_14, var_12);
            var_15 = wp::address(var_sensor_objid, var_4);
            var_17 = wp::load(var_15);
            var_16 = (var_13 == var_17);
            var_6 = var_6 && var_16;
        }
        if (var_6) {
            // adr = sensor_adr[sensorid]                                                         <L 1614>
            var_18 = wp::address(var_sensor_adr, var_4);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // sensordata_out[worldid, adr] += actuator_force_in[worldid, actid]                  <L 1615>
            var_21 = wp::address(var_actuator_force_in, var_0, var_2);
            var_23 = wp::load(var_21);
            var_22 = wp::atomic_add(var_sensordata_out, var_0, var_19, var_23);
        }
}



extern "C" __global__ void _limit_pos_b1ff6d76_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_limitpos_adr,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nl_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_pos_in,
    wp::array_t<wp::float32> var_efc_margin_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32 var_13;
        bool var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        bool var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        bool var_29;
        const wp::int32 var_30 = 3;
        bool var_31;
        const wp::int32 var_32 = 4;
        bool var_33;
        wp::float32* var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::array_t<wp::float32> var_39;
        //---------
        // forward
        // def _limit_pos(                                                                        <L 228>
        // worldid, efcid, limitposid = wp.tid()                                                  <L 247>
        builtin_tid3d(var_0, var_1, var_2);
        // ne = ne_in[worldid]                                                                    <L 249>
        var_3 = wp::address(var_ne_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // nf = nf_in[worldid]                                                                    <L 250>
        var_6 = wp::address(var_nf_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // nl = nl_in[worldid]                                                                    <L 251>
        var_9 = wp::address(var_nl_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if efcid < ne + nf or efcid >= ne + nf + nl:                                           <L 254>
        var_13 = wp::add(var_4, var_7);
        var_14 = (var_1 < var_13);
        var_12 = var_14;
        if (!var_12) {
            var_15 = wp::add(var_4, var_7);
            var_16 = wp::add(var_15, var_10);
            var_17 = (var_1 >= var_16);
            var_12 = var_12 || var_17;
        }
        if (var_12) {
            // return                                                                             <L 255>
            return;
        }
        // sensorid = sensor_limitpos_adr[limitposid]                                             <L 257>
        var_18 = wp::address(var_sensor_limitpos_adr, var_2);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if efc_id_in[worldid, efcid] == sensor_objid[sensorid]:                                <L 258>
        var_21 = wp::address(var_efc_id_in, var_0, var_1);
        var_22 = wp::address(var_sensor_objid, var_19);
        var_24 = wp::load(var_21);
        var_25 = wp::load(var_22);
        var_23 = (var_24 == var_25);
        if (var_23) {
            // efc_type = efc_type_in[worldid, efcid]                                             <L 259>
            var_26 = wp::address(var_efc_type_in, var_0, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // if efc_type == ConstraintType.LIMIT_JOINT or efc_type == ConstraintType.LIMIT_TENDON:       <L 260>
            var_31 = (var_27 == var_30);
            var_29 = var_31;
            if (!var_29) {
                var_33 = (var_27 == var_32);
                var_29 = var_29 || var_33;
            }
            if (var_29) {
                // val = efc_pos_in[worldid, efcid] - efc_margin_in[worldid, efcid]               <L 261>
                var_34 = wp::address(var_efc_pos_in, var_0, var_1);
                var_35 = wp::address(var_efc_margin_in, var_0, var_1);
                var_37 = wp::load(var_34);
                var_38 = wp::load(var_35);
                var_36 = wp::sub(var_37, var_38);
                // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, sensordata_out[worldid])       <L 262>
                var_39 = wp::view(var_sensordata_out, var_0);
                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_19, var_36, var_39);
            }
        }
}



extern "C" __global__ void _sensor_collision_bade267e_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_ngeom,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_pairid,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_contact_type_in,
    wp::array_t<wp::int32> var_contact_geomcollisionid_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_sensor_collision_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        const wp::int32 var_6 = 2;
        wp::int32 var_7;
        wp::int32 var_8;
        bool var_9;
        wp::vec_t<2, wp::int32>* var_10;
        wp::vec_t<2, wp::int32> var_11;
        wp::vec_t<2, wp::int32> var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        bool var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::vec_t<2, wp::int32>* var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::vec_t<2, wp::int32> var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::mat_t<3, 3, wp::float32>* var_45;
        wp::mat_t<3, 3, wp::float32> var_46;
        wp::mat_t<3, 3, wp::float32> var_47;
        const wp::int32 var_48 = 0;
        const wp::int32 var_49 = 0;
        wp::float32 var_50;
        const wp::int32 var_51 = 0;
        const wp::int32 var_52 = 1;
        wp::float32 var_53;
        const wp::int32 var_54 = 0;
        const wp::int32 var_55 = 2;
        wp::float32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        const wp::float32 var_58 = 0.5;
        wp::float32 var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        const wp::float32 var_62 = 0.5;
        wp::float32 var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        const wp::int32 var_66 = 0;
        const wp::int32 var_67 = 0;
        wp::float32 var_68;
        const wp::int32 var_69 = 1;
        const wp::int32 var_70 = 1;
        wp::float32 var_71;
        const wp::int32 var_72 = 2;
        const wp::int32 var_73 = 2;
        wp::float32 var_74;
        const wp::int32 var_75 = 3;
        const wp::int32 var_76 = 0;
        wp::float32 var_77;
        const wp::int32 var_78 = 4;
        const wp::int32 var_79 = 1;
        wp::float32 var_80;
        const wp::int32 var_81 = 5;
        const wp::int32 var_82 = 2;
        wp::float32 var_83;
        const wp::int32 var_84 = 6;
        //---------
        // forward
        // def _sensor_collision(                                                                 <L 759>
        // conid = wp.tid()                                                                       <L 775>
        var_0 = builtin_tid1d();
        // if conid >= nacon_in[0]:                                                               <L 777>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 778>
            return;
        }
        // if not contact_type_in[conid] & ContactType.SENSOR:                                    <L 780>
        var_5 = wp::address(var_contact_type_in, var_0);
        var_8 = wp::load(var_5);
        var_7 = wp::bit_and(var_8, var_6);
        var_9 = wp::unot(var_7);
        if (var_9) {
            // return                                                                             <L 781>
            return;
        }
        // geom = contact_geom_in[conid]                                                          <L 783>
        var_10 = wp::address(var_contact_geom_in, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if geom[0] <= geom[1]:                                                                 <L 784>
        var_14 = wp::extract(var_11, var_13);
        var_16 = wp::extract(var_11, var_15);
        var_17 = (var_14 <= var_16);
        if (var_17) {
            // pairid = math.upper_tri_index(ngeom, geom[0], geom[1])                             <L 785>
            var_19 = wp::extract(var_11, var_18);
            var_21 = wp::extract(var_11, var_20);
            var_22 = upper_tri_index_0(var_ngeom, var_19, var_21);
        }
        if (!var_17) {
            // pairid = math.upper_tri_index(ngeom, geom[1], geom[0])                             <L 787>
            var_24 = wp::extract(var_11, var_23);
            var_26 = wp::extract(var_11, var_25);
            var_27 = upper_tri_index_0(var_ngeom, var_24, var_26);
        }
        var_28 = wp::where(var_17, var_22, var_27);
        // worldid = contact_worldid_in[conid]                                                    <L 789>
        var_29 = wp::address(var_contact_worldid_in, var_0);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // collisionid = nxn_pairid[pairid][1]                                                    <L 790>
        var_32 = wp::address(var_nxn_pairid, var_28);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        // geomcollisionid = contact_geomcollisionid_in[conid]                                    <L 791>
        var_36 = wp::address(var_contact_geomcollisionid_in, var_0);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // dist = contact_dist_in[conid]                                                          <L 793>
        var_39 = wp::address(var_contact_dist_in, var_0);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // pos = contact_pos_in[conid]                                                            <L 794>
        var_42 = wp::address(var_contact_pos_in, var_0);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // frame = contact_frame_in[conid]                                                        <L 795>
        var_45 = wp::address(var_contact_frame_in, var_0);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // normal = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])                                <L 796>
        var_50 = wp::extract(var_46, var_48, var_49);
        var_53 = wp::extract(var_46, var_51, var_52);
        var_56 = wp::extract(var_46, var_54, var_55);
        var_57 = wp::vec_t<3, wp::float32>(var_50, var_53, var_56);
        // pnt1 = pos - 0.5 * dist * normal                                                       <L 797>
        var_59 = wp::mul(var_58, var_40);
        var_60 = wp::mul(var_59, var_57);
        var_61 = wp::sub(var_43, var_60);
        // pnt2 = pos + 0.5 * dist * normal                                                       <L 798>
        var_63 = wp::mul(var_62, var_40);
        var_64 = wp::mul(var_63, var_57);
        var_65 = wp::add(var_43, var_64);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 0] = dist                  <L 800>
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_66, var_40);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 1] = pnt1[0]               <L 801>
        var_68 = wp::extract(var_61, var_67);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_69, var_68);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 2] = pnt1[1]               <L 802>
        var_71 = wp::extract(var_61, var_70);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_72, var_71);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 3] = pnt1[2]               <L 803>
        var_74 = wp::extract(var_61, var_73);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_75, var_74);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 4] = pnt2[0]               <L 804>
        var_77 = wp::extract(var_65, var_76);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_78, var_77);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 5] = pnt2[1]               <L 805>
        var_80 = wp::extract(var_65, var_79);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_81, var_80);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 6] = pnt2[2]               <L 806>
        var_83 = wp::extract(var_65, var_82);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_84, var_83);
}



extern "C" __global__ void _limit_frc_be4f0b30_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_limitfrc_adr,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nl_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32 var_13;
        bool var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        bool var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        bool var_29;
        const wp::int32 var_30 = 3;
        bool var_31;
        const wp::int32 var_32 = 4;
        bool var_33;
        wp::float32* var_34;
        wp::array_t<wp::float32> var_35;
        wp::float32 var_36;
        //---------
        // forward
        // def _limit_frc(                                                                        <L 1640>
        // worldid, efcid, limitfrcid = wp.tid()                                                  <L 1658>
        builtin_tid3d(var_0, var_1, var_2);
        // ne = ne_in[worldid]                                                                    <L 1660>
        var_3 = wp::address(var_ne_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // nf = nf_in[worldid]                                                                    <L 1661>
        var_6 = wp::address(var_nf_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // nl = nl_in[worldid]                                                                    <L 1662>
        var_9 = wp::address(var_nl_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if efcid < ne + nf or efcid >= ne + nf + nl:                                           <L 1665>
        var_13 = wp::add(var_4, var_7);
        var_14 = (var_1 < var_13);
        var_12 = var_14;
        if (!var_12) {
            var_15 = wp::add(var_4, var_7);
            var_16 = wp::add(var_15, var_10);
            var_17 = (var_1 >= var_16);
            var_12 = var_12 || var_17;
        }
        if (var_12) {
            // return                                                                             <L 1666>
            return;
        }
        // sensorid = sensor_limitfrc_adr[limitfrcid]                                             <L 1668>
        var_18 = wp::address(var_sensor_limitfrc_adr, var_2);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if efc_id_in[worldid, efcid] == sensor_objid[sensorid]:                                <L 1669>
        var_21 = wp::address(var_efc_id_in, var_0, var_1);
        var_22 = wp::address(var_sensor_objid, var_19);
        var_24 = wp::load(var_21);
        var_25 = wp::load(var_22);
        var_23 = (var_24 == var_25);
        if (var_23) {
            // efc_type = efc_type_in[worldid, efcid]                                             <L 1670>
            var_26 = wp::address(var_efc_type_in, var_0, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // if efc_type == ConstraintType.LIMIT_JOINT or efc_type == ConstraintType.LIMIT_TENDON:       <L 1671>
            var_31 = (var_27 == var_30);
            var_29 = var_31;
            if (!var_29) {
                var_33 = (var_27 == var_32);
                var_29 = var_29 || var_33;
            }
            if (var_29) {
                // _write_scalar(                                                                 <L 1672>
                // sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, efc_force_in[worldid, efcid], sensordata_out[worldid]       <L 1673>
                var_34 = wp::address(var_efc_force_in, var_0, var_1);
                var_35 = wp::view(var_sensordata_out, var_0);
                var_36 = wp::load(var_34);
                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_19, var_36, var_35);
            }
        }
}



extern "C" __global__ void _limit_vel_ca76d496_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_limitvel_adr,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nl_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_vel_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32 var_13;
        bool var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        bool var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        bool var_29;
        const wp::int32 var_30 = 3;
        bool var_31;
        const wp::int32 var_32 = 4;
        bool var_33;
        wp::float32* var_34;
        wp::array_t<wp::float32> var_35;
        wp::float32 var_36;
        //---------
        // forward
        // def _limit_vel(                                                                        <L 1028>
        // worldid, efcid, limitvelid = wp.tid()                                                  <L 1046>
        builtin_tid3d(var_0, var_1, var_2);
        // ne = ne_in[worldid]                                                                    <L 1048>
        var_3 = wp::address(var_ne_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // nf = nf_in[worldid]                                                                    <L 1049>
        var_6 = wp::address(var_nf_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // nl = nl_in[worldid]                                                                    <L 1050>
        var_9 = wp::address(var_nl_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if efcid < ne + nf or efcid >= ne + nf + nl:                                           <L 1053>
        var_13 = wp::add(var_4, var_7);
        var_14 = (var_1 < var_13);
        var_12 = var_14;
        if (!var_12) {
            var_15 = wp::add(var_4, var_7);
            var_16 = wp::add(var_15, var_10);
            var_17 = (var_1 >= var_16);
            var_12 = var_12 || var_17;
        }
        if (var_12) {
            // return                                                                             <L 1054>
            return;
        }
        // sensorid = sensor_limitvel_adr[limitvelid]                                             <L 1056>
        var_18 = wp::address(var_sensor_limitvel_adr, var_2);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if efc_id_in[worldid, efcid] == sensor_objid[sensorid]:                                <L 1057>
        var_21 = wp::address(var_efc_id_in, var_0, var_1);
        var_22 = wp::address(var_sensor_objid, var_19);
        var_24 = wp::load(var_21);
        var_25 = wp::load(var_22);
        var_23 = (var_24 == var_25);
        if (var_23) {
            // efc_type = efc_type_in[worldid, efcid]                                             <L 1058>
            var_26 = wp::address(var_efc_type_in, var_0, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // if efc_type == ConstraintType.LIMIT_JOINT or efc_type == ConstraintType.LIMIT_TENDON:       <L 1059>
            var_31 = (var_27 == var_30);
            var_29 = var_31;
            if (!var_29) {
                var_33 = (var_27 == var_32);
                var_29 = var_29 || var_33;
            }
            if (var_29) {
                // _write_scalar(                                                                 <L 1060>
                // sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, efc_vel_in[worldid, efcid], sensordata_out[worldid]       <L 1061>
                var_34 = wp::address(var_efc_vel_in, var_0, var_1);
                var_35 = wp::view(var_sensordata_out, var_0);
                var_36 = wp::load(var_34);
                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_19, var_36, var_35);
            }
        }
}



extern "C" __global__ void _contact_match_d7b38672_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_opt_cone,
    wp::int32 var_opt_contact_sensor_maxmatch,
    bool var_opt_warn_overflow,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_size,
    wp::array_t<wp::int32> var_sensor_objtype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_reftype,
    wp::array_t<wp::int32> var_sensor_refid,
    wp::array_t<wp::int32> var_sensor_intprm,
    wp::array_t<wp::int32> var_sensor_contact_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_contact_type_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_overflow_out,
    wp::array_t<wp::int32> var_sensor_contact_nmatch_out,
    wp::array_t<wp::int32> var_sensor_contact_matchid_out,
    wp::array_t<wp::float32> var_sensor_contact_criteria_out,
    wp::array_t<wp::float32> var_sensor_contact_direction_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32* var_6;
        bool var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 6;
        bool var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::mat_t<3, 3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::int32* var_38;
        wp::vec_t<3, wp::float32>* var_39;
        bool var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::mat_t<3, 3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::int32 var_44;
        wp::vec_t<3, wp::float32> var_45;
        bool var_46;
        bool var_47;
        const wp::int32 var_48 = 0;
        bool var_49;
        const wp::int32 var_50 = 0;
        bool var_51;
        const wp::float32 var_52 = 1.0;
        wp::vec_t<2, wp::int32>* var_53;
        wp::vec_t<2, wp::int32> var_54;
        wp::vec_t<2, wp::int32> var_55;
        const wp::int32 var_56 = 0;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        bool var_66;
        bool var_67;
        bool var_68;
        bool var_69;
        bool var_70;
        bool var_71;
        bool var_72;
        bool var_73;
        bool var_74;
        bool var_75;
        const wp::float32 var_76 = 1.0;
        bool var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        const wp::int32 var_80 = 0;
        bool var_81;
        bool var_82;
        bool var_83;
        bool var_84;
        bool var_85;
        bool var_86;
        bool var_87;
        bool var_88;
        const wp::float32 var_89 = -1.0;
        wp::float32 var_90;
        const wp::int32 var_91 = 0;
        bool var_92;
        bool var_93;
        const wp::float32 var_94 = -1.0;
        wp::float32 var_95;
        const wp::int32 var_96 = 0;
        bool var_97;
        bool var_98;
        const wp::float32 var_99 = -1.0;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::array_t<wp::int32> var_105;
        const wp::int32 var_106 = 1;
        wp::int32 var_107;
        bool var_108;
        const wp::str var_109 = "contact match overflow: please increase Option.contact_sensor_maxmatch to %u\n";
        const wp::int32 var_110 = 64;
        const wp::int32 var_111 = 64;
        wp::int32 var_112;
        const wp::int32 var_113 = 1;
        bool var_114;
        wp::float32* var_115;
        wp::float32 var_116;
        const wp::int32 var_117 = 2;
        bool var_118;
        const bool var_119 = false;
        wp::vec_t<6, wp::float32> var_120;
        const wp::int32 var_121 = 0;
        wp::float32 var_122;
        const wp::int32 var_123 = 0;
        wp::float32 var_124;
        wp::float32 var_125;
        const wp::int32 var_126 = 1;
        wp::float32 var_127;
        const wp::int32 var_128 = 1;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        const wp::int32 var_132 = 2;
        wp::float32 var_133;
        const wp::int32 var_134 = 2;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        //---------
        // forward
        // def _contact_match(                                                                    <L 2331>
        // contactsensorid, contactid = wp.tid()                                                  <L 2369>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_contact_adr[contactsensorid]                                         <L 2370>
        var_2 = wp::address(var_sensor_contact_adr, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if contactid >= nacon_in[0]:                                                           <L 2372>
        var_6 = wp::address(var_nacon_in, var_5);
        var_8 = wp::load(var_6);
        var_7 = (var_1 >= var_8);
        if (var_7) {
            // return                                                                             <L 2373>
            return;
        }
        // if not contact_type_in[contactid] & ContactType.CONSTRAINT:                            <L 2375>
        var_9 = wp::address(var_contact_type_in, var_1);
        var_12 = wp::load(var_9);
        var_11 = wp::bit_and(var_12, var_10);
        var_13 = wp::unot(var_11);
        if (var_13) {
            // return                                                                             <L 2376>
            return;
        }
        // objtype = sensor_objtype[sensorid]                                                     <L 2379>
        var_14 = wp::address(var_sensor_objtype, var_3);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // objid = sensor_objid[sensorid]                                                         <L 2380>
        var_17 = wp::address(var_sensor_objid, var_3);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // reftype = sensor_reftype[sensorid]                                                     <L 2381>
        var_20 = wp::address(var_sensor_reftype, var_3);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // refid = sensor_refid[sensorid]                                                         <L 2382>
        var_23 = wp::address(var_sensor_refid, var_3);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // reduce = sensor_intprm[sensorid, 1]                                                    <L 2383>
        var_27 = wp::address(var_sensor_intprm, var_3, var_26);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // worldid = contact_worldid_in[contactid]                                                <L 2385>
        var_30 = wp::address(var_contact_worldid_in, var_1);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // if objtype == ObjType.SITE:                                                            <L 2388>
        var_34 = (var_15 == var_33);
        if (var_34) {
            // if not inside_geom(                                                                <L 2389>
            // site_xpos_in[worldid, objid], site_xmat_in[worldid, objid], site_size[objid], site_type[objid], contact_pos_in[contactid]       <L 2390>
            var_35 = wp::address(var_site_xpos_in, var_31, var_18);
            var_36 = wp::address(var_site_xmat_in, var_31, var_18);
            var_37 = wp::address(var_site_size, var_18);
            var_38 = wp::address(var_site_type, var_18);
            var_39 = wp::address(var_contact_pos_in, var_1);
            var_41 = wp::load(var_35);
            var_42 = wp::load(var_36);
            var_43 = wp::load(var_37);
            var_44 = wp::load(var_38);
            var_45 = wp::load(var_39);
            var_40 = inside_geom_0(var_41, var_42, var_43, var_44, var_45);
            var_46 = wp::unot(var_40);
            if (var_46) {
                // return                                                                         <L 2392>
                return;
            }
        }
        // if objtype == ObjType.UNKNOWN and reftype == ObjType.UNKNOWN:                          <L 2395>
        var_49 = (var_15 == var_48);
        var_47 = var_49;
        if (var_47) {
            var_51 = (var_21 == var_50);
            var_47 = var_47 && var_51;
        }
        if (var_47) {
            // dir = 1.0                                                                          <L 2396>
        }
        if (!var_47) {
            // geom = contact_geom_in[contactid]                                                  <L 2399>
            var_53 = wp::address(var_contact_geom_in, var_1);
            var_55 = wp::load(var_53);
            var_54 = wp::copy(var_55);
            // geom1 = geom[0]                                                                    <L 2400>
            var_57 = wp::extract(var_54, var_56);
            // geom2 = geom[1]                                                                    <L 2401>
            var_59 = wp::extract(var_54, var_58);
            // body1 = geom_bodyid[geom1]                                                         <L 2402>
            var_60 = wp::address(var_geom_bodyid, var_57);
            var_62 = wp::load(var_60);
            var_61 = wp::copy(var_62);
            // body2 = geom_bodyid[geom2]                                                         <L 2403>
            var_63 = wp::address(var_geom_bodyid, var_59);
            var_65 = wp::load(var_63);
            var_64 = wp::copy(var_65);
            // match11 = _check_match(body_parentid, body1, geom1, objtype, objid)                <L 2406>
            var_66 = _check_match_0(var_body_parentid, var_61, var_57, var_15, var_18);
            // match12 = _check_match(body_parentid, body2, geom2, objtype, objid)                <L 2407>
            var_67 = _check_match_0(var_body_parentid, var_64, var_59, var_15, var_18);
            // match21 = _check_match(body_parentid, body1, geom1, reftype, refid)                <L 2408>
            var_68 = _check_match_0(var_body_parentid, var_61, var_57, var_21, var_24);
            // match22 = _check_match(body_parentid, body2, geom2, reftype, refid)                <L 2409>
            var_69 = _check_match_0(var_body_parentid, var_64, var_59, var_21, var_24);
            // if not match11 and not match12:                                                    <L 2412>
            var_71 = wp::unot(var_66);
            var_70 = var_71;
            if (var_70) {
                var_72 = wp::unot(var_67);
                var_70 = var_70 && var_72;
            }
            if (var_70) {
                // return                                                                         <L 2413>
                return;
            }
            // if not match21 and not match22:                                                    <L 2414>
            var_74 = wp::unot(var_68);
            var_73 = var_74;
            if (var_73) {
                var_75 = wp::unot(var_69);
                var_73 = var_73 && var_75;
            }
            if (var_73) {
                // return                                                                         <L 2415>
                return;
            }
            // dir = 1.0                                                                          <L 2418>
            // if objtype != ObjType.UNKNOWN and reftype != ObjType.UNKNOWN:                      <L 2419>
            var_79 = (var_15 != var_78);
            var_77 = var_79;
            if (var_77) {
                var_81 = (var_21 != var_80);
                var_77 = var_77 && var_81;
            }
            if (var_77) {
                // order_regular = match11 and match22                                            <L 2421>
                var_82 = var_66;
                if (var_82) {
                    var_82 = var_82 && var_69;
                }
                // order_reverse = match12 and match21                                            <L 2422>
                var_83 = var_67;
                if (var_83) {
                    var_83 = var_83 && var_68;
                }
                // if not order_regular and not order_reverse:                                    <L 2423>
                var_85 = wp::unot(var_82);
                var_84 = var_85;
                if (var_84) {
                    var_86 = wp::unot(var_83);
                    var_84 = var_84 && var_86;
                }
                if (var_84) {
                    // return                                                                     <L 2424>
                    return;
                }
                // if order_reverse and not order_regular:                                        <L 2425>
                var_87 = var_83;
                if (var_87) {
                    var_88 = wp::unot(var_82);
                    var_87 = var_87 && var_88;
                }
                if (var_87) {
                    // dir = -1.0                                                                 <L 2426>
                }
                var_90 = wp::where(var_87, var_89, var_76);
            }
            if (!var_77) {
                // elif objtype != ObjType.UNKNOWN:                                               <L 2427>
                var_92 = (var_15 != var_91);
                if (var_92) {
                    // if not match11:                                                            <L 2428>
                    var_93 = wp::unot(var_66);
                    if (var_93) {
                        // dir = -1.0                                                             <L 2429>
                    }
                    var_95 = wp::where(var_93, var_94, var_76);
                }
                if (!var_92) {
                    // elif reftype != ObjType.UNKNOWN:                                           <L 2430>
                    var_97 = (var_21 != var_96);
                    if (var_97) {
                        // if not match22:                                                        <L 2431>
                        var_98 = wp::unot(var_69);
                        if (var_98) {
                            // dir = -1.0                                                         <L 2432>
                        }
                        var_100 = wp::where(var_98, var_99, var_76);
                    }
                    var_101 = wp::where(var_97, var_100, var_76);
                }
                var_102 = wp::where(var_92, var_95, var_101);
            }
            var_103 = wp::where(var_77, var_90, var_102);
        }
        var_104 = wp::where(var_47, var_52, var_103);
        // contactmatchid = wp.atomic_add(sensor_contact_nmatch_out[worldid], contactsensorid, 1)       <L 2434>
        var_105 = wp::view(var_sensor_contact_nmatch_out, var_31);
        var_107 = wp::atomic_add(var_105, var_0, var_106);
        // if contactmatchid >= opt_contact_sensor_maxmatch:                                      <L 2436>
        var_108 = (var_107 >= var_opt_contact_sensor_maxmatch);
        if (var_108) {
            // if opt_warn_overflow:                                                              <L 2437>
            if (var_opt_warn_overflow) {
                // wp.printf("contact match overflow: please increase Option.contact_sensor_maxmatch to %u\n", contactmatchid)       <L 2438>
                printf(var_109, var_107);
            }
            // wp.atomic_or(overflow_out, worldid, OverflowType.CONTACT_MATCH)                    <L 2439>
            var_112 = wp::atomic_or(var_overflow_out, var_31, var_111);
            // return                                                                             <L 2440>
            return;
        }
        // sensor_contact_matchid_out[worldid, contactsensorid, contactmatchid] = contactid       <L 2442>
        wp::array_store(var_sensor_contact_matchid_out, var_31, var_0, var_107, var_1);
        // if reduce == 1:  # mindist                                                             <L 2444>
        var_114 = (var_28 == var_113);
        if (var_114) {
            // sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = contact_dist_in[contactid]       <L 2445>
            var_115 = wp::address(var_contact_dist_in, var_1);
            var_116 = wp::load(var_115);
            wp::array_store(var_sensor_contact_criteria_out, var_31, var_0, var_107, var_116);
        }
        if (!var_114) {
            // elif reduce == 2:  # maxforce                                                      <L 2446>
            var_118 = (var_28 == var_117);
            if (var_118) {
                // contact_force = support.contact_force_fn(                                      <L 2447>
                // opt_cone,                                                                      <L 2448>
                // contact_frame_in,                                                              <L 2449>
                // contact_friction_in,                                                           <L 2450>
                // contact_dim_in,                                                                <L 2451>
                // contact_efc_address_in,                                                        <L 2452>
                // efc_force_in,                                                                  <L 2453>
                // njmax_in,                                                                      <L 2454>
                // nacon_in,                                                                      <L 2455>
                // worldid,                                                                       <L 2456>
                // contactid,                                                                     <L 2457>
                // False,                                                                         <L 2458>
                var_120 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_31, var_1, var_119);
                // force_magnitude = (                                                            <L 2460>
                // contact_force[0] * contact_force[0] + contact_force[1] * contact_force[1] + contact_force[2] * contact_force[2]       <L 2461>
                var_122 = wp::extract(var_120, var_121);
                var_124 = wp::extract(var_120, var_123);
                var_125 = wp::mul(var_122, var_124);
                var_127 = wp::extract(var_120, var_126);
                var_129 = wp::extract(var_120, var_128);
                var_130 = wp::mul(var_127, var_129);
                var_131 = wp::add(var_125, var_130);
                var_133 = wp::extract(var_120, var_132);
                var_135 = wp::extract(var_120, var_134);
                var_136 = wp::mul(var_133, var_135);
                var_137 = wp::add(var_131, var_136);
                // sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = -force_magnitude       <L 2463>
                var_138 = wp::neg(var_137);
                wp::array_store(var_sensor_contact_criteria_out, var_31, var_0, var_107, var_138);
            }
        }
        // sensor_contact_direction_out[worldid, contactsensorid, contactmatchid] = dir           <L 2466>
        wp::array_store(var_sensor_contact_direction_out, var_31, var_0, var_107, var_104);
}



extern "C" __global__ void _sensor_vel_e2a1adc2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objtype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_reftype,
    wp::array_t<wp::int32> var_sensor_refid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_vel_adr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::array_t<wp::float32> var_11;
        const wp::int32 var_12 = 2;
        bool var_13;
        wp::vec_t<3, wp::float32> var_14;
        const wp::int32 var_15 = 3;
        const wp::int32 var_16 = 3;
        bool var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 3;
        const wp::int32 var_20 = 10;
        bool var_21;
        wp::float32 var_22;
        const wp::int32 var_23 = 12;
        bool var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 14;
        bool var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 19;
        bool var_30;
        wp::vec_t<3, wp::float32> var_31;
        const wp::int32 var_32 = 3;
        const wp::int32 var_33 = 31;
        bool var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        const wp::int32 var_45 = 3;
        const wp::int32 var_46 = 32;
        bool var_47;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32* var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        const wp::int32 var_58 = 3;
        const wp::int32 var_59 = 36;
        bool var_60;
        wp::vec_t<3, wp::float32> var_61;
        const wp::int32 var_62 = 3;
        const wp::int32 var_63 = 37;
        bool var_64;
        wp::vec_t<3, wp::float32> var_65;
        const wp::int32 var_66 = 3;
        wp::vec_t<3, wp::float32> var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        //---------
        // forward
        // def _sensor_vel(                                                                       <L 1306>
        // worldid, velid = wp.tid()                                                              <L 1343>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_vel_adr[velid]                                                       <L 1344>
        var_2 = wp::address(var_sensor_vel_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // sensortype = sensor_type[sensorid]                                                     <L 1345>
        var_5 = wp::address(var_sensor_type, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // objid = sensor_objid[sensorid]                                                         <L 1346>
        var_8 = wp::address(var_sensor_objid, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // out = sensordata_out[worldid]                                                          <L 1347>
        var_11 = wp::view(var_sensordata_out, var_0);
        // if sensortype == SensorType.VELOCIMETER:                                               <L 1349>
        var_13 = (var_6 == var_12);
        if (var_13) {
            // vec3 = _velocimeter(body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cvel_in, worldid, objid)       <L 1350>
            var_14 = _velocimeter_0(var_body_rootid, var_site_bodyid, var_site_xpos_in, var_site_xmat_in, var_subtree_com_in, var_cvel_in, var_0, var_9);
            // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1351>
            _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_15, var_14, var_11);
        }
        if (!var_13) {
            // elif sensortype == SensorType.GYRO:                                                <L 1352>
            var_17 = (var_6 == var_16);
            if (var_17) {
                // vec3 = _gyro(site_bodyid, site_xmat_in, cvel_in, worldid, objid)               <L 1353>
                var_18 = _gyro_0(var_site_bodyid, var_site_xmat_in, var_cvel_in, var_0, var_9);
                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1354>
                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_19, var_18, var_11);
            }
            if (!var_17) {
                // elif sensortype == SensorType.JOINTVEL:                                        <L 1355>
                var_21 = (var_6 == var_20);
                if (var_21) {
                    // val = _joint_vel(jnt_dofadr, qvel_in, worldid, objid)                      <L 1356>
                    var_22 = _joint_vel_0(var_jnt_dofadr, var_qvel_in, var_0, var_9);
                    // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 1357>
                    _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_22, var_11);
                }
                if (!var_21) {
                    // elif sensortype == SensorType.TENDONVEL:                                   <L 1358>
                    var_24 = (var_6 == var_23);
                    if (var_24) {
                        // val = _tendon_vel(ten_velocity_in, worldid, objid)                     <L 1359>
                        var_25 = _tendon_vel_0(var_ten_velocity_in, var_0, var_9);
                        // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 1360>
                        _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_25, var_11);
                    }
                    if (!var_24) {
                        // elif sensortype == SensorType.ACTUATORVEL:                             <L 1361>
                        var_27 = (var_6 == var_26);
                        if (var_27) {
                            // val = _actuator_vel(actuator_velocity_in, worldid, objid)          <L 1362>
                            var_28 = _actuator_vel_0(var_actuator_velocity_in, var_0, var_9);
                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 1363>
                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_28, var_11);
                        }
                        if (!var_27) {
                            // elif sensortype == SensorType.BALLANGVEL:                          <L 1364>
                            var_30 = (var_6 == var_29);
                            if (var_30) {
                                // vec3 = _ball_ang_vel(jnt_dofadr, qvel_in, worldid, objid)       <L 1365>
                                var_31 = _ball_ang_vel_0(var_jnt_dofadr, var_qvel_in, var_0, var_9);
                                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1366>
                                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_32, var_31, var_11);
                            }
                            if (!var_30) {
                                // elif sensortype == SensorType.FRAMELINVEL:                     <L 1367>
                                var_34 = (var_6 == var_33);
                                if (var_34) {
                                    // objtype = sensor_objtype[sensorid]                         <L 1368>
                                    var_35 = wp::address(var_sensor_objtype, var_3);
                                    var_37 = wp::load(var_35);
                                    var_36 = wp::copy(var_37);
                                    // refid = sensor_refid[sensorid]                             <L 1369>
                                    var_38 = wp::address(var_sensor_refid, var_3);
                                    var_40 = wp::load(var_38);
                                    var_39 = wp::copy(var_40);
                                    // reftype = sensor_reftype[sensorid]                         <L 1370>
                                    var_41 = wp::address(var_sensor_reftype, var_3);
                                    var_43 = wp::load(var_41);
                                    var_42 = wp::copy(var_43);
                                    // frame_linvel = _frame_linvel(                              <L 1371>
                                    // body_rootid,                                               <L 1372>
                                    // geom_bodyid,                                               <L 1373>
                                    // site_bodyid,                                               <L 1374>
                                    // cam_bodyid,                                                <L 1375>
                                    // xpos_in,                                                   <L 1376>
                                    // xmat_in,                                                   <L 1377>
                                    // xipos_in,                                                  <L 1378>
                                    // ximat_in,                                                  <L 1379>
                                    // geom_xpos_in,                                              <L 1380>
                                    // geom_xmat_in,                                              <L 1381>
                                    // site_xpos_in,                                              <L 1382>
                                    // site_xmat_in,                                              <L 1383>
                                    // cam_xpos_in,                                               <L 1384>
                                    // cam_xmat_in,                                               <L 1385>
                                    // subtree_com_in,                                            <L 1386>
                                    // cvel_in,                                                   <L 1387>
                                    // worldid,                                                   <L 1388>
                                    // objid,                                                     <L 1389>
                                    // objtype,                                                   <L 1390>
                                    // refid,                                                     <L 1391>
                                    // reftype,                                                   <L 1392>
                                    var_44 = _frame_linvel_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xmat_in, var_xipos_in, var_ximat_in, var_geom_xpos_in, var_geom_xmat_in, var_site_xpos_in, var_site_xmat_in, var_cam_xpos_in, var_cam_xmat_in, var_subtree_com_in, var_cvel_in, var_0, var_9, var_36, var_39, var_42);
                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, frame_linvel, out)       <L 1394>
                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_45, var_44, var_11);
                                }
                                if (!var_34) {
                                    // elif sensortype == SensorType.FRAMEANGVEL:                 <L 1395>
                                    var_47 = (var_6 == var_46);
                                    if (var_47) {
                                        // objtype = sensor_objtype[sensorid]                     <L 1396>
                                        var_48 = wp::address(var_sensor_objtype, var_3);
                                        var_50 = wp::load(var_48);
                                        var_49 = wp::copy(var_50);
                                        // refid = sensor_refid[sensorid]                         <L 1397>
                                        var_51 = wp::address(var_sensor_refid, var_3);
                                        var_53 = wp::load(var_51);
                                        var_52 = wp::copy(var_53);
                                        // reftype = sensor_reftype[sensorid]                     <L 1398>
                                        var_54 = wp::address(var_sensor_reftype, var_3);
                                        var_56 = wp::load(var_54);
                                        var_55 = wp::copy(var_56);
                                        // frame_angvel = _frame_angvel(                          <L 1399>
                                        // body_rootid,                                           <L 1400>
                                        // geom_bodyid,                                           <L 1401>
                                        // site_bodyid,                                           <L 1402>
                                        // cam_bodyid,                                            <L 1403>
                                        // xpos_in,                                               <L 1404>
                                        // xmat_in,                                               <L 1405>
                                        // xipos_in,                                              <L 1406>
                                        // ximat_in,                                              <L 1407>
                                        // geom_xpos_in,                                          <L 1408>
                                        // geom_xmat_in,                                          <L 1409>
                                        // site_xpos_in,                                          <L 1410>
                                        // site_xmat_in,                                          <L 1411>
                                        // cam_xpos_in,                                           <L 1412>
                                        // cam_xmat_in,                                           <L 1413>
                                        // subtree_com_in,                                        <L 1414>
                                        // cvel_in,                                               <L 1415>
                                        // worldid,                                               <L 1416>
                                        // objid,                                                 <L 1417>
                                        // objtype,                                               <L 1418>
                                        // refid,                                                 <L 1419>
                                        // reftype,                                               <L 1420>
                                        var_57 = _frame_angvel_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xmat_in, var_xipos_in, var_ximat_in, var_geom_xpos_in, var_geom_xmat_in, var_site_xpos_in, var_site_xmat_in, var_cam_xpos_in, var_cam_xmat_in, var_subtree_com_in, var_cvel_in, var_0, var_9, var_49, var_52, var_55);
                                        // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, frame_angvel, out)       <L 1422>
                                        _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_58, var_57, var_11);
                                    }
                                    if (!var_47) {
                                        // elif sensortype == SensorType.SUBTREELINVEL:           <L 1423>
                                        var_60 = (var_6 == var_59);
                                        if (var_60) {
                                            // vec3 = _subtree_linvel(subtree_linvel_in, worldid, objid)       <L 1424>
                                            var_61 = _subtree_linvel_0(var_subtree_linvel_in, var_0, var_9);
                                            // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1425>
                                            _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_62, var_61, var_11);
                                        }
                                        if (!var_60) {
                                            // elif sensortype == SensorType.SUBTREEANGMOM:       <L 1426>
                                            var_64 = (var_6 == var_63);
                                            if (var_64) {
                                                // vec3 = _subtree_angmom(subtree_angmom_in, worldid, objid)       <L 1427>
                                                var_65 = _subtree_angmom_0(var_subtree_angmom_in, var_0, var_9);
                                                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1428>
                                                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_66, var_65, var_11);
                                            }
                                        }
                                        var_67 = wp::where(var_60, var_61, var_65);
                                    }
                                }
                                var_68 = wp::where(var_34, var_36, var_49);
                                var_69 = wp::where(var_34, var_39, var_52);
                                var_70 = wp::where(var_34, var_42, var_55);
                            }
                            var_71 = wp::where(var_30, var_31, var_67);
                        }
                    }
                    var_72 = wp::where(var_24, var_25, var_28);
                }
                var_73 = wp::where(var_21, var_22, var_72);
            }
            var_74 = wp::where(var_17, var_18, var_71);
        }
        var_75 = wp::where(var_13, var_14, var_74);
}



extern "C" __global__ void _sensor_pos_f481f6a4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_ngeom,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_magnetic,
    wp::array_t<wp::int32> var_body_geomnum,
    wp::array_t<wp::int32> var_body_geomadr,
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::int32> var_site_type,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_size,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::float32> var_cam_fovy,
    wp::array_t<wp::vec_t<2, wp::int32>> var_cam_resolution,
    wp::array_t<wp::vec_t<2, wp::float32>> var_cam_sensorsize,
    wp::array_t<wp::vec_t<4, wp::float32>> var_cam_intrinsic,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objtype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_reftype,
    wp::array_t<wp::int32> var_sensor_refid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_pairid,
    wp::array_t<wp::int32> var_sensor_pos_adr,
    wp::array_t<wp::int32> var_rangefinder_sensor_adr,
    wp::array_t<wp::float32> var_time_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_in,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::array_t<wp::float32> var_rangefinder_dist_in,
    wp::array_t<wp::float32> var_sensor_collision_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::array_t<wp::float32> var_11;
        const wp::int32 var_12 = 6;
        bool var_13;
        wp::vec_t<3, wp::float32> var_14;
        const wp::int32 var_15 = 3;
        const wp::int32 var_16 = 8;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::vec_t<2, wp::float32> var_21;
        const wp::int32 var_22 = 2;
        const wp::int32 var_23 = 7;
        bool var_24;
        wp::int32* var_25;
        wp::float32* var_26;
        wp::int32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 9;
        bool var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 11;
        bool var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 13;
        bool var_37;
        wp::float32 var_38;
        const wp::int32 var_39 = 18;
        bool var_40;
        wp::quat_t<wp::float32> var_41;
        const wp::int32 var_42 = 4;
        const wp::int32 var_43 = 26;
        bool var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32* var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::vec_t<3, wp::float32> var_54;
        const wp::int32 var_55 = 3;
        bool var_56;
        const wp::int32 var_57 = 28;
        bool var_58;
        const wp::int32 var_59 = 29;
        bool var_60;
        const wp::int32 var_61 = 30;
        bool var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        const wp::int32 var_72 = 28;
        bool var_73;
        const wp::int32 var_74 = 0;
        const wp::int32 var_75 = 29;
        bool var_76;
        const wp::int32 var_77 = 1;
        const wp::int32 var_78 = 30;
        bool var_79;
        const wp::int32 var_80 = 2;
        wp::int32 var_81;
        wp::int32 var_82;
        wp::vec_t<3, wp::float32> var_83;
        const wp::int32 var_84 = 3;
        const wp::int32 var_85 = 27;
        bool var_86;
        wp::int32* var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        wp::int32* var_90;
        wp::int32 var_91;
        wp::int32 var_92;
        wp::int32* var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        wp::quat_t<wp::float32> var_96;
        const wp::int32 var_97 = 4;
        const wp::int32 var_98 = 35;
        bool var_99;
        wp::vec_t<3, wp::float32> var_100;
        const wp::int32 var_101 = 3;
        bool var_102;
        const wp::int32 var_103 = 39;
        bool var_104;
        const wp::int32 var_105 = 40;
        bool var_106;
        const wp::int32 var_107 = 41;
        bool var_108;
        wp::int32* var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32* var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32* var_115;
        wp::int32 var_116;
        wp::int32 var_117;
        wp::int32* var_118;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::float32* var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::float32 var_124 = 0.0;
        const wp::float32 var_125 = 0.0;
        const wp::float32 var_126 = 0.0;
        const wp::float32 var_127 = 0.0;
        const wp::float32 var_128 = 0.0;
        const wp::float32 var_129 = 0.0;
        wp::vec_t<6, wp::float32> var_130;
        const bool var_131 = false;
        bool var_132;
        const wp::int32 var_133 = 1;
        const wp::int32 var_134 = 1;
        wp::int32 var_135;
        bool var_136;
        wp::int32* var_137;
        wp::int32 var_138;
        wp::int32 var_139;
        wp::int32* var_140;
        wp::int32 var_141;
        wp::int32 var_142;
        const wp::int32 var_143 = 1;
        wp::int32 var_144;
        wp::int32 var_145;
        wp::int32 var_146;
        const wp::int32 var_147 = 1;
        const wp::int32 var_148 = 1;
        wp::int32 var_149;
        bool var_150;
        wp::int32* var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::int32* var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        const wp::int32 var_157 = 1;
        wp::int32 var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        wp::range_t var_161;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::range_t var_164;
        wp::int32 var_165;
        wp::int32 var_166;
        bool var_167;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::int32 var_170;
        wp::vec_t<2, wp::int32>* var_171;
        const wp::int32 var_172 = 1;
        wp::int32 var_173;
        wp::vec_t<2, wp::int32> var_174;
        const wp::int32 var_175 = 0;
        const wp::int32 var_176 = 0;
        wp::float32* var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        bool var_180;
        wp::float32 var_181;
        bool var_182;
        const wp::int32 var_183 = 40;
        bool var_184;
        const wp::int32 var_185 = 41;
        bool var_186;
        const wp::int32 var_187 = 1;
        wp::float32* var_188;
        const wp::int32 var_189 = 2;
        wp::float32* var_190;
        const wp::int32 var_191 = 3;
        wp::float32* var_192;
        const wp::int32 var_193 = 4;
        wp::float32* var_194;
        const wp::int32 var_195 = 5;
        wp::float32* var_196;
        const wp::int32 var_197 = 6;
        wp::float32* var_198;
        wp::vec_t<6, wp::float32> var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        wp::float32 var_202;
        wp::float32 var_203;
        wp::float32 var_204;
        wp::float32 var_205;
        wp::vec_t<6, wp::float32> var_206;
        wp::int32* var_207;
        wp::int32* var_208;
        bool var_209;
        wp::int32 var_210;
        wp::int32 var_211;
        const bool var_212 = true;
        wp::int32* var_213;
        wp::int32* var_214;
        bool var_215;
        wp::int32 var_216;
        wp::int32 var_217;
        bool var_218;
        const bool var_219 = false;
        bool var_220;
        bool var_221;
        wp::float32 var_222;
        wp::vec_t<6, wp::float32> var_223;
        bool var_224;
        const wp::int32 var_225 = 1;
        const wp::int32 var_226 = 0;
        wp::float32* var_227;
        wp::float32 var_228;
        wp::float32 var_229;
        bool var_230;
        wp::float32 var_231;
        bool var_232;
        const wp::int32 var_233 = 40;
        bool var_234;
        const wp::int32 var_235 = 41;
        bool var_236;
        const wp::int32 var_237 = 1;
        wp::float32* var_238;
        const wp::int32 var_239 = 2;
        wp::float32* var_240;
        const wp::int32 var_241 = 3;
        wp::float32* var_242;
        const wp::int32 var_243 = 4;
        wp::float32* var_244;
        const wp::int32 var_245 = 5;
        wp::float32* var_246;
        const wp::int32 var_247 = 6;
        wp::float32* var_248;
        wp::vec_t<6, wp::float32> var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        wp::float32 var_252;
        wp::float32 var_253;
        wp::float32 var_254;
        wp::float32 var_255;
        wp::vec_t<6, wp::float32> var_256;
        wp::int32* var_257;
        wp::int32* var_258;
        bool var_259;
        wp::int32 var_260;
        wp::int32 var_261;
        const bool var_262 = true;
        wp::int32* var_263;
        wp::int32* var_264;
        bool var_265;
        wp::int32 var_266;
        wp::int32 var_267;
        bool var_268;
        const bool var_269 = false;
        bool var_270;
        bool var_271;
        wp::float32 var_272;
        wp::vec_t<6, wp::float32> var_273;
        bool var_274;
        const wp::int32 var_275 = 2;
        const wp::int32 var_276 = 0;
        wp::float32* var_277;
        wp::float32 var_278;
        wp::float32 var_279;
        bool var_280;
        wp::float32 var_281;
        bool var_282;
        const wp::int32 var_283 = 40;
        bool var_284;
        const wp::int32 var_285 = 41;
        bool var_286;
        const wp::int32 var_287 = 1;
        wp::float32* var_288;
        const wp::int32 var_289 = 2;
        wp::float32* var_290;
        const wp::int32 var_291 = 3;
        wp::float32* var_292;
        const wp::int32 var_293 = 4;
        wp::float32* var_294;
        const wp::int32 var_295 = 5;
        wp::float32* var_296;
        const wp::int32 var_297 = 6;
        wp::float32* var_298;
        wp::vec_t<6, wp::float32> var_299;
        wp::float32 var_300;
        wp::float32 var_301;
        wp::float32 var_302;
        wp::float32 var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::vec_t<6, wp::float32> var_306;
        wp::int32* var_307;
        wp::int32* var_308;
        bool var_309;
        wp::int32 var_310;
        wp::int32 var_311;
        const bool var_312 = true;
        wp::int32* var_313;
        wp::int32* var_314;
        bool var_315;
        wp::int32 var_316;
        wp::int32 var_317;
        bool var_318;
        const bool var_319 = false;
        bool var_320;
        bool var_321;
        wp::float32 var_322;
        wp::vec_t<6, wp::float32> var_323;
        bool var_324;
        const wp::int32 var_325 = 3;
        const wp::int32 var_326 = 0;
        wp::float32* var_327;
        wp::float32 var_328;
        wp::float32 var_329;
        bool var_330;
        wp::float32 var_331;
        bool var_332;
        const wp::int32 var_333 = 40;
        bool var_334;
        const wp::int32 var_335 = 41;
        bool var_336;
        const wp::int32 var_337 = 1;
        wp::float32* var_338;
        const wp::int32 var_339 = 2;
        wp::float32* var_340;
        const wp::int32 var_341 = 3;
        wp::float32* var_342;
        const wp::int32 var_343 = 4;
        wp::float32* var_344;
        const wp::int32 var_345 = 5;
        wp::float32* var_346;
        const wp::int32 var_347 = 6;
        wp::float32* var_348;
        wp::vec_t<6, wp::float32> var_349;
        wp::float32 var_350;
        wp::float32 var_351;
        wp::float32 var_352;
        wp::float32 var_353;
        wp::float32 var_354;
        wp::float32 var_355;
        wp::vec_t<6, wp::float32> var_356;
        wp::int32* var_357;
        wp::int32* var_358;
        bool var_359;
        wp::int32 var_360;
        wp::int32 var_361;
        const bool var_362 = true;
        wp::int32* var_363;
        wp::int32* var_364;
        bool var_365;
        wp::int32 var_366;
        wp::int32 var_367;
        bool var_368;
        const bool var_369 = false;
        bool var_370;
        bool var_371;
        wp::float32 var_372;
        wp::vec_t<6, wp::float32> var_373;
        bool var_374;
        const wp::int32 var_375 = 4;
        const wp::int32 var_376 = 0;
        wp::float32* var_377;
        wp::float32 var_378;
        wp::float32 var_379;
        bool var_380;
        wp::float32 var_381;
        bool var_382;
        const wp::int32 var_383 = 40;
        bool var_384;
        const wp::int32 var_385 = 41;
        bool var_386;
        const wp::int32 var_387 = 1;
        wp::float32* var_388;
        const wp::int32 var_389 = 2;
        wp::float32* var_390;
        const wp::int32 var_391 = 3;
        wp::float32* var_392;
        const wp::int32 var_393 = 4;
        wp::float32* var_394;
        const wp::int32 var_395 = 5;
        wp::float32* var_396;
        const wp::int32 var_397 = 6;
        wp::float32* var_398;
        wp::vec_t<6, wp::float32> var_399;
        wp::float32 var_400;
        wp::float32 var_401;
        wp::float32 var_402;
        wp::float32 var_403;
        wp::float32 var_404;
        wp::float32 var_405;
        wp::vec_t<6, wp::float32> var_406;
        wp::int32* var_407;
        wp::int32* var_408;
        bool var_409;
        wp::int32 var_410;
        wp::int32 var_411;
        const bool var_412 = true;
        wp::int32* var_413;
        wp::int32* var_414;
        bool var_415;
        wp::int32 var_416;
        wp::int32 var_417;
        bool var_418;
        const bool var_419 = false;
        bool var_420;
        bool var_421;
        wp::float32 var_422;
        wp::vec_t<6, wp::float32> var_423;
        bool var_424;
        const wp::int32 var_425 = 5;
        const wp::int32 var_426 = 0;
        wp::float32* var_427;
        wp::float32 var_428;
        wp::float32 var_429;
        bool var_430;
        wp::float32 var_431;
        bool var_432;
        const wp::int32 var_433 = 40;
        bool var_434;
        const wp::int32 var_435 = 41;
        bool var_436;
        const wp::int32 var_437 = 1;
        wp::float32* var_438;
        const wp::int32 var_439 = 2;
        wp::float32* var_440;
        const wp::int32 var_441 = 3;
        wp::float32* var_442;
        const wp::int32 var_443 = 4;
        wp::float32* var_444;
        const wp::int32 var_445 = 5;
        wp::float32* var_446;
        const wp::int32 var_447 = 6;
        wp::float32* var_448;
        wp::vec_t<6, wp::float32> var_449;
        wp::float32 var_450;
        wp::float32 var_451;
        wp::float32 var_452;
        wp::float32 var_453;
        wp::float32 var_454;
        wp::float32 var_455;
        wp::vec_t<6, wp::float32> var_456;
        wp::int32* var_457;
        wp::int32* var_458;
        bool var_459;
        wp::int32 var_460;
        wp::int32 var_461;
        const bool var_462 = true;
        wp::int32* var_463;
        wp::int32* var_464;
        bool var_465;
        wp::int32 var_466;
        wp::int32 var_467;
        bool var_468;
        const bool var_469 = false;
        bool var_470;
        bool var_471;
        wp::float32 var_472;
        wp::vec_t<6, wp::float32> var_473;
        bool var_474;
        const wp::int32 var_475 = 6;
        const wp::int32 var_476 = 0;
        wp::float32* var_477;
        wp::float32 var_478;
        wp::float32 var_479;
        bool var_480;
        wp::float32 var_481;
        bool var_482;
        const wp::int32 var_483 = 40;
        bool var_484;
        const wp::int32 var_485 = 41;
        bool var_486;
        const wp::int32 var_487 = 1;
        wp::float32* var_488;
        const wp::int32 var_489 = 2;
        wp::float32* var_490;
        const wp::int32 var_491 = 3;
        wp::float32* var_492;
        const wp::int32 var_493 = 4;
        wp::float32* var_494;
        const wp::int32 var_495 = 5;
        wp::float32* var_496;
        const wp::int32 var_497 = 6;
        wp::float32* var_498;
        wp::vec_t<6, wp::float32> var_499;
        wp::float32 var_500;
        wp::float32 var_501;
        wp::float32 var_502;
        wp::float32 var_503;
        wp::float32 var_504;
        wp::float32 var_505;
        wp::vec_t<6, wp::float32> var_506;
        wp::int32* var_507;
        wp::int32* var_508;
        bool var_509;
        wp::int32 var_510;
        wp::int32 var_511;
        const bool var_512 = true;
        wp::int32* var_513;
        wp::int32* var_514;
        bool var_515;
        wp::int32 var_516;
        wp::int32 var_517;
        bool var_518;
        const bool var_519 = false;
        bool var_520;
        bool var_521;
        wp::float32 var_522;
        wp::vec_t<6, wp::float32> var_523;
        bool var_524;
        const wp::int32 var_525 = 7;
        const wp::int32 var_526 = 0;
        wp::float32* var_527;
        wp::float32 var_528;
        wp::float32 var_529;
        bool var_530;
        wp::float32 var_531;
        bool var_532;
        const wp::int32 var_533 = 40;
        bool var_534;
        const wp::int32 var_535 = 41;
        bool var_536;
        const wp::int32 var_537 = 1;
        wp::float32* var_538;
        const wp::int32 var_539 = 2;
        wp::float32* var_540;
        const wp::int32 var_541 = 3;
        wp::float32* var_542;
        const wp::int32 var_543 = 4;
        wp::float32* var_544;
        const wp::int32 var_545 = 5;
        wp::float32* var_546;
        const wp::int32 var_547 = 6;
        wp::float32* var_548;
        wp::vec_t<6, wp::float32> var_549;
        wp::float32 var_550;
        wp::float32 var_551;
        wp::float32 var_552;
        wp::float32 var_553;
        wp::float32 var_554;
        wp::float32 var_555;
        wp::vec_t<6, wp::float32> var_556;
        wp::int32* var_557;
        wp::int32* var_558;
        bool var_559;
        wp::int32 var_560;
        wp::int32 var_561;
        const bool var_562 = true;
        wp::int32* var_563;
        wp::int32* var_564;
        bool var_565;
        wp::int32 var_566;
        wp::int32 var_567;
        bool var_568;
        const bool var_569 = false;
        bool var_570;
        bool var_571;
        wp::float32 var_572;
        wp::vec_t<6, wp::float32> var_573;
        bool var_574;
        const wp::int32 var_575 = 39;
        const wp::int32 var_576 = 39;
        wp::int32 var_577;
        bool var_578;
        const wp::int32 var_579 = 40;
        const wp::int32 var_580 = 40;
        wp::int32 var_581;
        bool var_582;
        wp::float32* var_583;
        bool var_584;
        wp::float32 var_585;
        const wp::int32 var_586 = 3;
        wp::float32 var_587;
        const wp::int32 var_588 = 0;
        wp::float32 var_589;
        wp::float32 var_590;
        const wp::int32 var_591 = 4;
        wp::float32 var_592;
        const wp::int32 var_593 = 1;
        wp::float32 var_594;
        wp::float32 var_595;
        const wp::int32 var_596 = 5;
        wp::float32 var_597;
        const wp::int32 var_598 = 2;
        wp::float32 var_599;
        wp::float32 var_600;
        wp::vec_t<3, wp::float32> var_601;
        wp::vec_t<3, wp::float32> var_602;
        const wp::float32 var_603 = -1.0;
        wp::vec_t<3, wp::float32> var_604;
        wp::vec_t<3, wp::float32> var_605;
        const wp::float32 var_606 = 0.0;
        const wp::float32 var_607 = 0.0;
        const wp::float32 var_608 = 0.0;
        wp::vec_t<3, wp::float32> var_609;
        wp::vec_t<3, wp::float32> var_610;
        const wp::int32 var_611 = 3;
        const wp::int32 var_612 = 41;
        const wp::int32 var_613 = 41;
        wp::int32 var_614;
        bool var_615;
        wp::float32* var_616;
        bool var_617;
        wp::float32 var_618;
        const wp::int32 var_619 = 3;
        wp::float32 var_620;
        const wp::int32 var_621 = 4;
        wp::float32 var_622;
        const wp::int32 var_623 = 5;
        wp::float32 var_624;
        const wp::int32 var_625 = 0;
        wp::float32 var_626;
        const wp::int32 var_627 = 1;
        wp::float32 var_628;
        const wp::int32 var_629 = 2;
        wp::float32 var_630;
        wp::vec_t<6, wp::float32> var_631;
        wp::vec_t<6, wp::float32> var_632;
        wp::vec_t<6, wp::float32> var_633;
        const wp::float32 var_634 = 0.0;
        const wp::float32 var_635 = 0.0;
        const wp::float32 var_636 = 0.0;
        const wp::float32 var_637 = 0.0;
        const wp::float32 var_638 = 0.0;
        const wp::float32 var_639 = 0.0;
        wp::vec_t<6, wp::float32> var_640;
        wp::vec_t<6, wp::float32> var_641;
        const wp::int32 var_642 = 6;
        const wp::int32 var_643 = 38;
        bool var_644;
        wp::int32* var_645;
        wp::int32 var_646;
        wp::int32 var_647;
        const wp::int32 var_648 = 2;
        bool var_649;
        wp::vec_t<3, wp::float32>* var_650;
        wp::vec_t<3, wp::float32> var_651;
        wp::vec_t<3, wp::float32> var_652;
        const wp::int32 var_653 = 1;
        bool var_654;
        const wp::int32 var_655 = 0;
        bool var_656;
        bool var_657;
        wp::shape_t* var_658;
        const wp::int32 var_659 = 0;
        wp::int32 var_660;
        wp::shape_t var_661;
        wp::int32 var_662;
        wp::float32* var_663;
        const wp::float32 var_664 = 1e-15;
        bool var_665;
        wp::float32 var_666;
        wp::shape_t* var_667;
        const wp::int32 var_668 = 0;
        wp::int32 var_669;
        wp::shape_t var_670;
        wp::int32 var_671;
        wp::float32* var_672;
        bool var_673;
        wp::float32 var_674;
        wp::vec_t<3, wp::float32>* var_675;
        wp::vec_t<3, wp::float32> var_676;
        wp::vec_t<3, wp::float32> var_677;
        wp::vec_t<3, wp::float32>* var_678;
        wp::vec_t<3, wp::float32> var_679;
        wp::vec_t<3, wp::float32> var_680;
        wp::vec_t<3, wp::float32> var_681;
        wp::vec_t<3, wp::float32>* var_682;
        wp::vec_t<3, wp::float32> var_683;
        wp::vec_t<3, wp::float32> var_684;
        wp::vec_t<3, wp::float32> var_685;
        const wp::int32 var_686 = 5;
        bool var_687;
        wp::vec_t<3, wp::float32>* var_688;
        wp::vec_t<3, wp::float32> var_689;
        wp::vec_t<3, wp::float32> var_690;
        const wp::int32 var_691 = 6;
        bool var_692;
        wp::vec_t<3, wp::float32>* var_693;
        wp::vec_t<3, wp::float32> var_694;
        wp::vec_t<3, wp::float32> var_695;
        const wp::int32 var_696 = 7;
        bool var_697;
        wp::vec_t<3, wp::float32>* var_698;
        wp::vec_t<3, wp::float32> var_699;
        wp::vec_t<3, wp::float32> var_700;
        wp::vec_t<3, wp::float32> var_701;
        wp::vec_t<3, wp::float32> var_702;
        wp::vec_t<3, wp::float32> var_703;
        wp::vec_t<3, wp::float32> var_704;
        wp::int32* var_705;
        wp::int32 var_706;
        wp::int32 var_707;
        wp::vec_t<3, wp::float32>* var_708;
        wp::mat_t<3, 3, wp::float32>* var_709;
        wp::vec_t<3, wp::float32>* var_710;
        wp::int32* var_711;
        bool var_712;
        wp::vec_t<3, wp::float32> var_713;
        wp::mat_t<3, 3, wp::float32> var_714;
        wp::vec_t<3, wp::float32> var_715;
        wp::int32 var_716;
        wp::float32 var_717;
        const wp::int32 var_718 = 43;
        bool var_719;
        wp::vec_t<2, wp::float32>* var_720;
        const wp::int32 var_721 = 0;
        wp::float32 var_722;
        wp::vec_t<2, wp::float32> var_723;
        const wp::int32 var_724 = 44;
        bool var_725;
        wp::vec_t<2, wp::float32>* var_726;
        const wp::int32 var_727 = 1;
        wp::float32 var_728;
        wp::vec_t<2, wp::float32> var_729;
        const wp::int32 var_730 = 45;
        bool var_731;
        wp::float32 var_732;
        wp::float32 var_733;
        wp::float32 var_734;
        wp::int32 var_735;
        wp::int32 var_736;
        wp::int32 var_737;
        wp::int32 var_738;
        wp::int32 var_739;
        wp::int32 var_740;
        wp::int32 var_741;
        wp::int32 var_742;
        wp::int32 var_743;
        wp::int32 var_744;
        wp::int32 var_745;
        wp::int32 var_746;
        wp::vec_t<3, wp::float32> var_747;
        wp::int32 var_748;
        wp::int32 var_749;
        wp::int32 var_750;
        wp::int32 var_751;
        wp::vec_t<3, wp::float32> var_752;
        wp::int32 var_753;
        wp::quat_t<wp::float32> var_754;
        wp::int32 var_755;
        wp::float32 var_756;
        wp::int32 var_757;
        wp::float32 var_758;
        wp::int32 var_759;
        wp::float32 var_760;
        wp::int32 var_761;
        wp::float32 var_762;
        wp::int32 var_763;
        wp::int32 var_764;
        wp::int32 var_765;
        wp::vec_t<3, wp::float32> var_766;
        //---------
        // forward
        // def _sensor_pos(                                                                       <L 495>
        // worldid, posid = wp.tid()                                                              <L 553>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_pos_adr[posid]                                                       <L 554>
        var_2 = wp::address(var_sensor_pos_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // sensortype = sensor_type[sensorid]                                                     <L 555>
        var_5 = wp::address(var_sensor_type, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // objid = sensor_objid[sensorid]                                                         <L 556>
        var_8 = wp::address(var_sensor_objid, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // out = sensordata_out[worldid]                                                          <L 557>
        var_11 = wp::view(var_sensordata_out, var_0);
        // if sensortype == SensorType.MAGNETOMETER:                                              <L 559>
        var_13 = (var_6 == var_12);
        if (var_13) {
            // vec3 = _magnetometer(opt_magnetic, site_xmat_in, worldid, objid)                   <L 560>
            var_14 = _magnetometer_0(var_opt_magnetic, var_site_xmat_in, var_0, var_9);
            // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 561>
            _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_15, var_14, var_11);
        }
        if (!var_13) {
            // elif sensortype == SensorType.CAMPROJECTION:                                       <L 562>
            var_17 = (var_6 == var_16);
            if (var_17) {
                // refid = sensor_refid[sensorid]                                                 <L 563>
                var_18 = wp::address(var_sensor_refid, var_3);
                var_20 = wp::load(var_18);
                var_19 = wp::copy(var_20);
                // vec2 = _cam_projection(                                                        <L 564>
                // cam_fovy, cam_resolution, cam_sensorsize, cam_intrinsic, site_xpos_in, cam_xpos_in, cam_xmat_in, worldid, objid, refid       <L 565>
                var_21 = _cam_projection_0(var_cam_fovy, var_cam_resolution, var_cam_sensorsize, var_cam_intrinsic, var_site_xpos_in, var_cam_xpos_in, var_cam_xmat_in, var_0, var_9, var_19);
                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 2, vec2, out)       <L 567>
                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_22, var_21, var_11);
            }
            if (!var_17) {
                // elif sensortype == SensorType.RANGEFINDER:                                     <L 568>
                var_24 = (var_6 == var_23);
                if (var_24) {
                    // val = rangefinder_dist_in[worldid, rangefinder_sensor_adr[sensorid]]       <L 569>
                    var_25 = wp::address(var_rangefinder_sensor_adr, var_3);
                    var_27 = wp::load(var_25);
                    var_26 = wp::address(var_rangefinder_dist_in, var_0, var_27);
                    var_29 = wp::load(var_26);
                    var_28 = wp::copy(var_29);
                    // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 570>
                    _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_28, var_11);
                }
                if (!var_24) {
                    // elif sensortype == SensorType.JOINTPOS:                                    <L 571>
                    var_31 = (var_6 == var_30);
                    if (var_31) {
                        // val = _joint_pos(jnt_qposadr, qpos_in, worldid, objid)                 <L 572>
                        var_32 = _joint_pos_0(var_jnt_qposadr, var_qpos_in, var_0, var_9);
                        // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 573>
                        _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_32, var_11);
                    }
                    if (!var_31) {
                        // elif sensortype == SensorType.TENDONPOS:                               <L 574>
                        var_34 = (var_6 == var_33);
                        if (var_34) {
                            // val = _tendon_pos(ten_length_in, worldid, objid)                   <L 575>
                            var_35 = _tendon_pos_0(var_ten_length_in, var_0, var_9);
                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 576>
                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_35, var_11);
                        }
                        if (!var_34) {
                            // elif sensortype == SensorType.ACTUATORPOS:                         <L 577>
                            var_37 = (var_6 == var_36);
                            if (var_37) {
                                // val = _actuator_pos(actuator_length_in, worldid, objid)        <L 578>
                                var_38 = _actuator_pos_0(var_actuator_length_in, var_0, var_9);
                                // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 579>
                                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_38, var_11);
                            }
                            if (!var_37) {
                                // elif sensortype == SensorType.BALLQUAT:                        <L 580>
                                var_40 = (var_6 == var_39);
                                if (var_40) {
                                    // quat = _ball_quat(jnt_qposadr, qpos_in, worldid, objid)       <L 581>
                                    var_41 = _ball_quat_0(var_jnt_qposadr, var_qpos_in, var_0, var_9);
                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 4, quat, out)       <L 582>
                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_42, var_41, var_11);
                                }
                                if (!var_40) {
                                    // elif sensortype == SensorType.FRAMEPOS:                    <L 583>
                                    var_44 = (var_6 == var_43);
                                    if (var_44) {
                                        // objtype = sensor_objtype[sensorid]                     <L 584>
                                        var_45 = wp::address(var_sensor_objtype, var_3);
                                        var_47 = wp::load(var_45);
                                        var_46 = wp::copy(var_47);
                                        // refid = sensor_refid[sensorid]                         <L 585>
                                        var_48 = wp::address(var_sensor_refid, var_3);
                                        var_50 = wp::load(var_48);
                                        var_49 = wp::copy(var_50);
                                        // reftype = sensor_reftype[sensorid]                     <L 586>
                                        var_51 = wp::address(var_sensor_reftype, var_3);
                                        var_53 = wp::load(var_51);
                                        var_52 = wp::copy(var_53);
                                        // vec3 = _frame_pos(                                     <L 587>
                                        // xpos_in,                                               <L 588>
                                        // xmat_in,                                               <L 589>
                                        // xipos_in,                                              <L 590>
                                        // ximat_in,                                              <L 591>
                                        // geom_xpos_in,                                          <L 592>
                                        // geom_xmat_in,                                          <L 593>
                                        // site_xpos_in,                                          <L 594>
                                        // site_xmat_in,                                          <L 595>
                                        // cam_xpos_in,                                           <L 596>
                                        // cam_xmat_in,                                           <L 597>
                                        // worldid,                                               <L 598>
                                        // objid,                                                 <L 599>
                                        // objtype,                                               <L 600>
                                        // refid,                                                 <L 601>
                                        // reftype,                                               <L 602>
                                        var_54 = _frame_pos_0(var_xpos_in, var_xmat_in, var_xipos_in, var_ximat_in, var_geom_xpos_in, var_geom_xmat_in, var_site_xpos_in, var_site_xmat_in, var_cam_xpos_in, var_cam_xmat_in, var_0, var_9, var_46, var_49, var_52);
                                        // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 604>
                                        _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_55, var_54, var_11);
                                    }
                                    if (!var_44) {
                                        // elif sensortype == SensorType.FRAMEXAXIS or sensortype == SensorType.FRAMEYAXIS or sensortype == SensorType.FRAMEZAXIS:       <L 605>
                                        var_58 = (var_6 == var_57);
                                        var_56 = var_58;
                                        if (!var_56) {
                                            var_60 = (var_6 == var_59);
                                            var_56 = var_56 || var_60;
                                        }
                                        if (!var_56) {
                                            var_62 = (var_6 == var_61);
                                            var_56 = var_56 || var_62;
                                        }
                                        if (var_56) {
                                            // objtype = sensor_objtype[sensorid]                 <L 606>
                                            var_63 = wp::address(var_sensor_objtype, var_3);
                                            var_65 = wp::load(var_63);
                                            var_64 = wp::copy(var_65);
                                            // refid = sensor_refid[sensorid]                     <L 607>
                                            var_66 = wp::address(var_sensor_refid, var_3);
                                            var_68 = wp::load(var_66);
                                            var_67 = wp::copy(var_68);
                                            // reftype = sensor_reftype[sensorid]                 <L 608>
                                            var_69 = wp::address(var_sensor_reftype, var_3);
                                            var_71 = wp::load(var_69);
                                            var_70 = wp::copy(var_71);
                                            // if sensortype == SensorType.FRAMEXAXIS:            <L 609>
                                            var_73 = (var_6 == var_72);
                                            if (var_73) {
                                                // axis = 0                                       <L 610>
                                            }
                                            if (!var_73) {
                                                // elif sensortype == SensorType.FRAMEYAXIS:       <L 611>
                                                var_76 = (var_6 == var_75);
                                                if (var_76) {
                                                    // axis = 1                                   <L 612>
                                                }
                                                if (!var_76) {
                                                    // elif sensortype == SensorType.FRAMEZAXIS:       <L 613>
                                                    var_79 = (var_6 == var_78);
                                                    if (var_79) {
                                                        // axis = 2                               <L 614>
                                                    }
                                                }
                                                var_81 = wp::where(var_76, var_77, var_80);
                                            }
                                            var_82 = wp::where(var_73, var_74, var_81);
                                            // vec3 = _frame_axis(                                <L 615>
                                            // xmat_in, ximat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, objid, objtype, refid, reftype, axis       <L 616>
                                            var_83 = _frame_axis_0(var_xmat_in, var_ximat_in, var_geom_xmat_in, var_site_xmat_in, var_cam_xmat_in, var_0, var_9, var_64, var_67, var_70, var_82);
                                            // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 618>
                                            _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_84, var_83, var_11);
                                        }
                                        if (!var_56) {
                                            // elif sensortype == SensorType.FRAMEQUAT:           <L 619>
                                            var_86 = (var_6 == var_85);
                                            if (var_86) {
                                                // objtype = sensor_objtype[sensorid]             <L 620>
                                                var_87 = wp::address(var_sensor_objtype, var_3);
                                                var_89 = wp::load(var_87);
                                                var_88 = wp::copy(var_89);
                                                // refid = sensor_refid[sensorid]                 <L 621>
                                                var_90 = wp::address(var_sensor_refid, var_3);
                                                var_92 = wp::load(var_90);
                                                var_91 = wp::copy(var_92);
                                                // reftype = sensor_reftype[sensorid]             <L 622>
                                                var_93 = wp::address(var_sensor_reftype, var_3);
                                                var_95 = wp::load(var_93);
                                                var_94 = wp::copy(var_95);
                                                // quat = _frame_quat(                            <L 623>
                                                // body_iquat,                                    <L 624>
                                                // geom_bodyid,                                   <L 625>
                                                // geom_quat,                                     <L 626>
                                                // site_bodyid,                                   <L 627>
                                                // site_quat,                                     <L 628>
                                                // cam_bodyid,                                    <L 629>
                                                // cam_quat,                                      <L 630>
                                                // xquat_in,                                      <L 631>
                                                // worldid,                                       <L 632>
                                                // objid,                                         <L 633>
                                                // objtype,                                       <L 634>
                                                // refid,                                         <L 635>
                                                // reftype,                                       <L 636>
                                                var_96 = _frame_quat_0(var_body_iquat, var_geom_bodyid, var_geom_quat, var_site_bodyid, var_site_quat, var_cam_bodyid, var_cam_quat, var_xquat_in, var_0, var_9, var_88, var_91, var_94);
                                                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 4, quat, out)       <L 638>
                                                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_97, var_96, var_11);
                                            }
                                            if (!var_86) {
                                                // elif sensortype == SensorType.SUBTREECOM:       <L 639>
                                                var_99 = (var_6 == var_98);
                                                if (var_99) {
                                                    // vec3 = _subtree_com(subtree_com_in, worldid, objid)       <L 640>
                                                    var_100 = _subtree_com_0(var_subtree_com_in, var_0, var_9);
                                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 641>
                                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_101, var_100, var_11);
                                                }
                                                if (!var_99) {
                                                    // elif sensortype == SensorType.GEOMDIST or sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 642>
                                                    var_104 = (var_6 == var_103);
                                                    var_102 = var_104;
                                                    if (!var_102) {
                                                        var_106 = (var_6 == var_105);
                                                        var_102 = var_102 || var_106;
                                                    }
                                                    if (!var_102) {
                                                        var_108 = (var_6 == var_107);
                                                        var_102 = var_102 || var_108;
                                                    }
                                                    if (var_102) {
                                                        // objtype = sensor_objtype[sensorid]       <L 643>
                                                        var_109 = wp::address(var_sensor_objtype, var_3);
                                                        var_111 = wp::load(var_109);
                                                        var_110 = wp::copy(var_111);
                                                        // objid = sensor_objid[sensorid]         <L 644>
                                                        var_112 = wp::address(var_sensor_objid, var_3);
                                                        var_114 = wp::load(var_112);
                                                        var_113 = wp::copy(var_114);
                                                        // reftype = sensor_reftype[sensorid]       <L 645>
                                                        var_115 = wp::address(var_sensor_reftype, var_3);
                                                        var_117 = wp::load(var_115);
                                                        var_116 = wp::copy(var_117);
                                                        // refid = sensor_refid[sensorid]         <L 646>
                                                        var_118 = wp::address(var_sensor_refid, var_3);
                                                        var_120 = wp::load(var_118);
                                                        var_119 = wp::copy(var_120);
                                                        // dist = float(sensor_cutoff[sensorid])       <L 649>
                                                        var_121 = wp::address(var_sensor_cutoff, var_3);
                                                        var_123 = wp::load(var_121);
                                                        var_122 = wp::float(var_123);
                                                        // pnts = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 650>
                                                        var_130 = wp::vec_t<6, wp::float32>({var_124, var_125, var_126, var_127, var_128, var_129});
                                                        // flip = bool(False)                     <L 651>
                                                        var_132 = bool(var_131);
                                                        // if objtype == int(ObjType.BODY.value):       <L 654>
                                                        var_135 = wp::int(var_134);
                                                        var_136 = (var_110 == var_135);
                                                        if (var_136) {
                                                            // n1 = body_geomnum[objid]           <L 655>
                                                            var_137 = wp::address(var_body_geomnum, var_113);
                                                            var_139 = wp::load(var_137);
                                                            var_138 = wp::copy(var_139);
                                                            // id1 = body_geomadr[objid]          <L 656>
                                                            var_140 = wp::address(var_body_geomadr, var_113);
                                                            var_142 = wp::load(var_140);
                                                            var_141 = wp::copy(var_142);
                                                        }
                                                        if (!var_136) {
                                                            // n1 = 1                             <L 658>
                                                            // id1 = objid                        <L 659>
                                                            var_144 = wp::copy(var_113);
                                                        }
                                                        var_145 = wp::where(var_136, var_138, var_143);
                                                        var_146 = wp::where(var_136, var_141, var_144);
                                                        // if reftype == int(ObjType.BODY.value):       <L 660>
                                                        var_149 = wp::int(var_148);
                                                        var_150 = (var_116 == var_149);
                                                        if (var_150) {
                                                            // n2 = body_geomnum[refid]           <L 661>
                                                            var_151 = wp::address(var_body_geomnum, var_119);
                                                            var_153 = wp::load(var_151);
                                                            var_152 = wp::copy(var_153);
                                                            // id2 = body_geomadr[refid]          <L 662>
                                                            var_154 = wp::address(var_body_geomadr, var_119);
                                                            var_156 = wp::load(var_154);
                                                            var_155 = wp::copy(var_156);
                                                        }
                                                        if (!var_150) {
                                                            // n2 = 1                             <L 664>
                                                            // id2 = refid                        <L 665>
                                                            var_158 = wp::copy(var_119);
                                                        }
                                                        var_159 = wp::where(var_150, var_152, var_157);
                                                        var_160 = wp::where(var_150, var_155, var_158);
                                                        // for geom1 in range(n1):                <L 667>
                                                        var_161 = wp::range(var_145);
                                                        start_for_0:;
                                                            if (iter_cmp(var_161) == 0) goto end_for_0;
                                                            var_162 = wp::iter_next(var_161);
                                                            // geomid1 = id1 + geom1              <L 668>
                                                            var_163 = wp::add(var_146, var_162);
                                                            // for geom2 in range(n2):            <L 669>
                                                            var_164 = wp::range(var_159);
                                                            start_for_2:;
                                                                if (iter_cmp(var_164) == 0) goto end_for_2;
                                                                var_165 = wp::iter_next(var_164);
                                                                // geomid2 = id2 + geom2          <L 670>
                                                                var_166 = wp::add(var_160, var_165);
                                                                // if geomid1 <= geomid2:         <L 672>
                                                                var_167 = (var_163 <= var_166);
                                                                if (var_167) {
                                                                    // pairid = math.upper_tri_index(ngeom, geomid1, geomid2)       <L 673>
                                                                    var_168 = upper_tri_index_0(var_ngeom, var_163, var_166);
                                                                }
                                                                if (!var_167) {
                                                                    // pairid = math.upper_tri_index(ngeom, geomid2, geomid1)       <L 675>
                                                                    var_169 = upper_tri_index_0(var_ngeom, var_166, var_163);
                                                                }
                                                                var_170 = wp::where(var_167, var_168, var_169);
                                                                // collisionid = nxn_pairid[pairid][1]       <L 676>
                                                                var_171 = wp::address(var_nxn_pairid, var_170);
                                                                var_174 = wp::load(var_171);
                                                                var_173 = wp::extract(var_174, var_172);
                                                                // for i in range(8):             <L 678>
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 679>
                                                                var_177 = wp::address(var_sensor_collision_in, var_0, var_173, var_175, var_176);
                                                                var_179 = wp::load(var_177);
                                                                var_178 = wp::copy(var_179);
                                                                // if dist_new < dist:            <L 681>
                                                                var_180 = (var_178 < var_122);
                                                                if (var_180) {
                                                                    // dist = dist_new            <L 682>
                                                                    var_181 = wp::copy(var_178);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 684>
                                                                    var_184 = (var_6 == var_183);
                                                                    var_182 = var_184;
                                                                    if (!var_182) {
                                                                        var_186 = (var_6 == var_185);
                                                                        var_182 = var_182 || var_186;
                                                                    }
                                                                    if (var_182) {
                                                                        // pnts = vec6(           <L 685>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 686>
                                                                        var_188 = wp::address(var_sensor_collision_in, var_0, var_173, var_175, var_187);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 687>
                                                                        var_190 = wp::address(var_sensor_collision_in, var_0, var_173, var_175, var_189);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 688>
                                                                        var_192 = wp::address(var_sensor_collision_in, var_0, var_173, var_175, var_191);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 689>
                                                                        var_194 = wp::address(var_sensor_collision_in, var_0, var_173, var_175, var_193);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 690>
                                                                        var_196 = wp::address(var_sensor_collision_in, var_0, var_173, var_175, var_195);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 691>
                                                                        var_198 = wp::address(var_sensor_collision_in, var_0, var_173, var_175, var_197);
                                                                        var_200 = wp::load(var_188);
                                                                        var_201 = wp::load(var_190);
                                                                        var_202 = wp::load(var_192);
                                                                        var_203 = wp::load(var_194);
                                                                        var_204 = wp::load(var_196);
                                                                        var_205 = wp::load(var_198);
                                                                        var_199 = wp::vec_t<6, wp::float32>({var_200, var_201, var_202, var_203, var_204, var_205});
                                                                    }
                                                                    var_206 = wp::where(var_182, var_199, var_130);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 694>
                                                                    var_207 = wp::address(var_geom_type, var_163);
                                                                    var_208 = wp::address(var_geom_type, var_166);
                                                                    var_210 = wp::load(var_207);
                                                                    var_211 = wp::load(var_208);
                                                                    var_209 = (var_210 > var_211);
                                                                    if (var_209) {
                                                                        // flip = True            <L 695>
                                                                    }
                                                                    if (!var_209) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 696>
                                                                        var_213 = wp::address(var_geom_type, var_163);
                                                                        var_214 = wp::address(var_geom_type, var_166);
                                                                        var_216 = wp::load(var_213);
                                                                        var_217 = wp::load(var_214);
                                                                        var_215 = (var_216 == var_217);
                                                                        if (var_215) {
                                                                            // flip = geomid1 > geomid2       <L 697>
                                                                            var_218 = (var_163 > var_166);
                                                                        }
                                                                        if (!var_215) {
                                                                            // flip = False       <L 699>
                                                                        }
                                                                        var_220 = wp::where(var_215, var_218, var_219);
                                                                    }
                                                                    var_221 = wp::where(var_209, var_212, var_220);
                                                                }
                                                                var_222 = wp::where(var_180, var_181, var_122);
                                                                var_223 = wp::where(var_180, var_206, var_130);
                                                                var_224 = wp::where(var_180, var_221, var_132);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 679>
                                                                var_227 = wp::address(var_sensor_collision_in, var_0, var_173, var_225, var_226);
                                                                var_229 = wp::load(var_227);
                                                                var_228 = wp::copy(var_229);
                                                                // if dist_new < dist:            <L 681>
                                                                var_230 = (var_228 < var_222);
                                                                if (var_230) {
                                                                    // dist = dist_new            <L 682>
                                                                    var_231 = wp::copy(var_228);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 684>
                                                                    var_234 = (var_6 == var_233);
                                                                    var_232 = var_234;
                                                                    if (!var_232) {
                                                                        var_236 = (var_6 == var_235);
                                                                        var_232 = var_232 || var_236;
                                                                    }
                                                                    if (var_232) {
                                                                        // pnts = vec6(           <L 685>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 686>
                                                                        var_238 = wp::address(var_sensor_collision_in, var_0, var_173, var_225, var_237);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 687>
                                                                        var_240 = wp::address(var_sensor_collision_in, var_0, var_173, var_225, var_239);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 688>
                                                                        var_242 = wp::address(var_sensor_collision_in, var_0, var_173, var_225, var_241);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 689>
                                                                        var_244 = wp::address(var_sensor_collision_in, var_0, var_173, var_225, var_243);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 690>
                                                                        var_246 = wp::address(var_sensor_collision_in, var_0, var_173, var_225, var_245);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 691>
                                                                        var_248 = wp::address(var_sensor_collision_in, var_0, var_173, var_225, var_247);
                                                                        var_250 = wp::load(var_238);
                                                                        var_251 = wp::load(var_240);
                                                                        var_252 = wp::load(var_242);
                                                                        var_253 = wp::load(var_244);
                                                                        var_254 = wp::load(var_246);
                                                                        var_255 = wp::load(var_248);
                                                                        var_249 = wp::vec_t<6, wp::float32>({var_250, var_251, var_252, var_253, var_254, var_255});
                                                                    }
                                                                    var_256 = wp::where(var_232, var_249, var_223);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 694>
                                                                    var_257 = wp::address(var_geom_type, var_163);
                                                                    var_258 = wp::address(var_geom_type, var_166);
                                                                    var_260 = wp::load(var_257);
                                                                    var_261 = wp::load(var_258);
                                                                    var_259 = (var_260 > var_261);
                                                                    if (var_259) {
                                                                        // flip = True            <L 695>
                                                                    }
                                                                    if (!var_259) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 696>
                                                                        var_263 = wp::address(var_geom_type, var_163);
                                                                        var_264 = wp::address(var_geom_type, var_166);
                                                                        var_266 = wp::load(var_263);
                                                                        var_267 = wp::load(var_264);
                                                                        var_265 = (var_266 == var_267);
                                                                        if (var_265) {
                                                                            // flip = geomid1 > geomid2       <L 697>
                                                                            var_268 = (var_163 > var_166);
                                                                        }
                                                                        if (!var_265) {
                                                                            // flip = False       <L 699>
                                                                        }
                                                                        var_270 = wp::where(var_265, var_268, var_269);
                                                                    }
                                                                    var_271 = wp::where(var_259, var_262, var_270);
                                                                }
                                                                var_272 = wp::where(var_230, var_231, var_222);
                                                                var_273 = wp::where(var_230, var_256, var_223);
                                                                var_274 = wp::where(var_230, var_271, var_224);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 679>
                                                                var_277 = wp::address(var_sensor_collision_in, var_0, var_173, var_275, var_276);
                                                                var_279 = wp::load(var_277);
                                                                var_278 = wp::copy(var_279);
                                                                // if dist_new < dist:            <L 681>
                                                                var_280 = (var_278 < var_272);
                                                                if (var_280) {
                                                                    // dist = dist_new            <L 682>
                                                                    var_281 = wp::copy(var_278);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 684>
                                                                    var_284 = (var_6 == var_283);
                                                                    var_282 = var_284;
                                                                    if (!var_282) {
                                                                        var_286 = (var_6 == var_285);
                                                                        var_282 = var_282 || var_286;
                                                                    }
                                                                    if (var_282) {
                                                                        // pnts = vec6(           <L 685>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 686>
                                                                        var_288 = wp::address(var_sensor_collision_in, var_0, var_173, var_275, var_287);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 687>
                                                                        var_290 = wp::address(var_sensor_collision_in, var_0, var_173, var_275, var_289);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 688>
                                                                        var_292 = wp::address(var_sensor_collision_in, var_0, var_173, var_275, var_291);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 689>
                                                                        var_294 = wp::address(var_sensor_collision_in, var_0, var_173, var_275, var_293);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 690>
                                                                        var_296 = wp::address(var_sensor_collision_in, var_0, var_173, var_275, var_295);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 691>
                                                                        var_298 = wp::address(var_sensor_collision_in, var_0, var_173, var_275, var_297);
                                                                        var_300 = wp::load(var_288);
                                                                        var_301 = wp::load(var_290);
                                                                        var_302 = wp::load(var_292);
                                                                        var_303 = wp::load(var_294);
                                                                        var_304 = wp::load(var_296);
                                                                        var_305 = wp::load(var_298);
                                                                        var_299 = wp::vec_t<6, wp::float32>({var_300, var_301, var_302, var_303, var_304, var_305});
                                                                    }
                                                                    var_306 = wp::where(var_282, var_299, var_273);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 694>
                                                                    var_307 = wp::address(var_geom_type, var_163);
                                                                    var_308 = wp::address(var_geom_type, var_166);
                                                                    var_310 = wp::load(var_307);
                                                                    var_311 = wp::load(var_308);
                                                                    var_309 = (var_310 > var_311);
                                                                    if (var_309) {
                                                                        // flip = True            <L 695>
                                                                    }
                                                                    if (!var_309) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 696>
                                                                        var_313 = wp::address(var_geom_type, var_163);
                                                                        var_314 = wp::address(var_geom_type, var_166);
                                                                        var_316 = wp::load(var_313);
                                                                        var_317 = wp::load(var_314);
                                                                        var_315 = (var_316 == var_317);
                                                                        if (var_315) {
                                                                            // flip = geomid1 > geomid2       <L 697>
                                                                            var_318 = (var_163 > var_166);
                                                                        }
                                                                        if (!var_315) {
                                                                            // flip = False       <L 699>
                                                                        }
                                                                        var_320 = wp::where(var_315, var_318, var_319);
                                                                    }
                                                                    var_321 = wp::where(var_309, var_312, var_320);
                                                                }
                                                                var_322 = wp::where(var_280, var_281, var_272);
                                                                var_323 = wp::where(var_280, var_306, var_273);
                                                                var_324 = wp::where(var_280, var_321, var_274);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 679>
                                                                var_327 = wp::address(var_sensor_collision_in, var_0, var_173, var_325, var_326);
                                                                var_329 = wp::load(var_327);
                                                                var_328 = wp::copy(var_329);
                                                                // if dist_new < dist:            <L 681>
                                                                var_330 = (var_328 < var_322);
                                                                if (var_330) {
                                                                    // dist = dist_new            <L 682>
                                                                    var_331 = wp::copy(var_328);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 684>
                                                                    var_334 = (var_6 == var_333);
                                                                    var_332 = var_334;
                                                                    if (!var_332) {
                                                                        var_336 = (var_6 == var_335);
                                                                        var_332 = var_332 || var_336;
                                                                    }
                                                                    if (var_332) {
                                                                        // pnts = vec6(           <L 685>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 686>
                                                                        var_338 = wp::address(var_sensor_collision_in, var_0, var_173, var_325, var_337);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 687>
                                                                        var_340 = wp::address(var_sensor_collision_in, var_0, var_173, var_325, var_339);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 688>
                                                                        var_342 = wp::address(var_sensor_collision_in, var_0, var_173, var_325, var_341);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 689>
                                                                        var_344 = wp::address(var_sensor_collision_in, var_0, var_173, var_325, var_343);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 690>
                                                                        var_346 = wp::address(var_sensor_collision_in, var_0, var_173, var_325, var_345);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 691>
                                                                        var_348 = wp::address(var_sensor_collision_in, var_0, var_173, var_325, var_347);
                                                                        var_350 = wp::load(var_338);
                                                                        var_351 = wp::load(var_340);
                                                                        var_352 = wp::load(var_342);
                                                                        var_353 = wp::load(var_344);
                                                                        var_354 = wp::load(var_346);
                                                                        var_355 = wp::load(var_348);
                                                                        var_349 = wp::vec_t<6, wp::float32>({var_350, var_351, var_352, var_353, var_354, var_355});
                                                                    }
                                                                    var_356 = wp::where(var_332, var_349, var_323);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 694>
                                                                    var_357 = wp::address(var_geom_type, var_163);
                                                                    var_358 = wp::address(var_geom_type, var_166);
                                                                    var_360 = wp::load(var_357);
                                                                    var_361 = wp::load(var_358);
                                                                    var_359 = (var_360 > var_361);
                                                                    if (var_359) {
                                                                        // flip = True            <L 695>
                                                                    }
                                                                    if (!var_359) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 696>
                                                                        var_363 = wp::address(var_geom_type, var_163);
                                                                        var_364 = wp::address(var_geom_type, var_166);
                                                                        var_366 = wp::load(var_363);
                                                                        var_367 = wp::load(var_364);
                                                                        var_365 = (var_366 == var_367);
                                                                        if (var_365) {
                                                                            // flip = geomid1 > geomid2       <L 697>
                                                                            var_368 = (var_163 > var_166);
                                                                        }
                                                                        if (!var_365) {
                                                                            // flip = False       <L 699>
                                                                        }
                                                                        var_370 = wp::where(var_365, var_368, var_369);
                                                                    }
                                                                    var_371 = wp::where(var_359, var_362, var_370);
                                                                }
                                                                var_372 = wp::where(var_330, var_331, var_322);
                                                                var_373 = wp::where(var_330, var_356, var_323);
                                                                var_374 = wp::where(var_330, var_371, var_324);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 679>
                                                                var_377 = wp::address(var_sensor_collision_in, var_0, var_173, var_375, var_376);
                                                                var_379 = wp::load(var_377);
                                                                var_378 = wp::copy(var_379);
                                                                // if dist_new < dist:            <L 681>
                                                                var_380 = (var_378 < var_372);
                                                                if (var_380) {
                                                                    // dist = dist_new            <L 682>
                                                                    var_381 = wp::copy(var_378);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 684>
                                                                    var_384 = (var_6 == var_383);
                                                                    var_382 = var_384;
                                                                    if (!var_382) {
                                                                        var_386 = (var_6 == var_385);
                                                                        var_382 = var_382 || var_386;
                                                                    }
                                                                    if (var_382) {
                                                                        // pnts = vec6(           <L 685>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 686>
                                                                        var_388 = wp::address(var_sensor_collision_in, var_0, var_173, var_375, var_387);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 687>
                                                                        var_390 = wp::address(var_sensor_collision_in, var_0, var_173, var_375, var_389);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 688>
                                                                        var_392 = wp::address(var_sensor_collision_in, var_0, var_173, var_375, var_391);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 689>
                                                                        var_394 = wp::address(var_sensor_collision_in, var_0, var_173, var_375, var_393);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 690>
                                                                        var_396 = wp::address(var_sensor_collision_in, var_0, var_173, var_375, var_395);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 691>
                                                                        var_398 = wp::address(var_sensor_collision_in, var_0, var_173, var_375, var_397);
                                                                        var_400 = wp::load(var_388);
                                                                        var_401 = wp::load(var_390);
                                                                        var_402 = wp::load(var_392);
                                                                        var_403 = wp::load(var_394);
                                                                        var_404 = wp::load(var_396);
                                                                        var_405 = wp::load(var_398);
                                                                        var_399 = wp::vec_t<6, wp::float32>({var_400, var_401, var_402, var_403, var_404, var_405});
                                                                    }
                                                                    var_406 = wp::where(var_382, var_399, var_373);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 694>
                                                                    var_407 = wp::address(var_geom_type, var_163);
                                                                    var_408 = wp::address(var_geom_type, var_166);
                                                                    var_410 = wp::load(var_407);
                                                                    var_411 = wp::load(var_408);
                                                                    var_409 = (var_410 > var_411);
                                                                    if (var_409) {
                                                                        // flip = True            <L 695>
                                                                    }
                                                                    if (!var_409) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 696>
                                                                        var_413 = wp::address(var_geom_type, var_163);
                                                                        var_414 = wp::address(var_geom_type, var_166);
                                                                        var_416 = wp::load(var_413);
                                                                        var_417 = wp::load(var_414);
                                                                        var_415 = (var_416 == var_417);
                                                                        if (var_415) {
                                                                            // flip = geomid1 > geomid2       <L 697>
                                                                            var_418 = (var_163 > var_166);
                                                                        }
                                                                        if (!var_415) {
                                                                            // flip = False       <L 699>
                                                                        }
                                                                        var_420 = wp::where(var_415, var_418, var_419);
                                                                    }
                                                                    var_421 = wp::where(var_409, var_412, var_420);
                                                                }
                                                                var_422 = wp::where(var_380, var_381, var_372);
                                                                var_423 = wp::where(var_380, var_406, var_373);
                                                                var_424 = wp::where(var_380, var_421, var_374);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 679>
                                                                var_427 = wp::address(var_sensor_collision_in, var_0, var_173, var_425, var_426);
                                                                var_429 = wp::load(var_427);
                                                                var_428 = wp::copy(var_429);
                                                                // if dist_new < dist:            <L 681>
                                                                var_430 = (var_428 < var_422);
                                                                if (var_430) {
                                                                    // dist = dist_new            <L 682>
                                                                    var_431 = wp::copy(var_428);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 684>
                                                                    var_434 = (var_6 == var_433);
                                                                    var_432 = var_434;
                                                                    if (!var_432) {
                                                                        var_436 = (var_6 == var_435);
                                                                        var_432 = var_432 || var_436;
                                                                    }
                                                                    if (var_432) {
                                                                        // pnts = vec6(           <L 685>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 686>
                                                                        var_438 = wp::address(var_sensor_collision_in, var_0, var_173, var_425, var_437);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 687>
                                                                        var_440 = wp::address(var_sensor_collision_in, var_0, var_173, var_425, var_439);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 688>
                                                                        var_442 = wp::address(var_sensor_collision_in, var_0, var_173, var_425, var_441);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 689>
                                                                        var_444 = wp::address(var_sensor_collision_in, var_0, var_173, var_425, var_443);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 690>
                                                                        var_446 = wp::address(var_sensor_collision_in, var_0, var_173, var_425, var_445);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 691>
                                                                        var_448 = wp::address(var_sensor_collision_in, var_0, var_173, var_425, var_447);
                                                                        var_450 = wp::load(var_438);
                                                                        var_451 = wp::load(var_440);
                                                                        var_452 = wp::load(var_442);
                                                                        var_453 = wp::load(var_444);
                                                                        var_454 = wp::load(var_446);
                                                                        var_455 = wp::load(var_448);
                                                                        var_449 = wp::vec_t<6, wp::float32>({var_450, var_451, var_452, var_453, var_454, var_455});
                                                                    }
                                                                    var_456 = wp::where(var_432, var_449, var_423);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 694>
                                                                    var_457 = wp::address(var_geom_type, var_163);
                                                                    var_458 = wp::address(var_geom_type, var_166);
                                                                    var_460 = wp::load(var_457);
                                                                    var_461 = wp::load(var_458);
                                                                    var_459 = (var_460 > var_461);
                                                                    if (var_459) {
                                                                        // flip = True            <L 695>
                                                                    }
                                                                    if (!var_459) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 696>
                                                                        var_463 = wp::address(var_geom_type, var_163);
                                                                        var_464 = wp::address(var_geom_type, var_166);
                                                                        var_466 = wp::load(var_463);
                                                                        var_467 = wp::load(var_464);
                                                                        var_465 = (var_466 == var_467);
                                                                        if (var_465) {
                                                                            // flip = geomid1 > geomid2       <L 697>
                                                                            var_468 = (var_163 > var_166);
                                                                        }
                                                                        if (!var_465) {
                                                                            // flip = False       <L 699>
                                                                        }
                                                                        var_470 = wp::where(var_465, var_468, var_469);
                                                                    }
                                                                    var_471 = wp::where(var_459, var_462, var_470);
                                                                }
                                                                var_472 = wp::where(var_430, var_431, var_422);
                                                                var_473 = wp::where(var_430, var_456, var_423);
                                                                var_474 = wp::where(var_430, var_471, var_424);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 679>
                                                                var_477 = wp::address(var_sensor_collision_in, var_0, var_173, var_475, var_476);
                                                                var_479 = wp::load(var_477);
                                                                var_478 = wp::copy(var_479);
                                                                // if dist_new < dist:            <L 681>
                                                                var_480 = (var_478 < var_472);
                                                                if (var_480) {
                                                                    // dist = dist_new            <L 682>
                                                                    var_481 = wp::copy(var_478);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 684>
                                                                    var_484 = (var_6 == var_483);
                                                                    var_482 = var_484;
                                                                    if (!var_482) {
                                                                        var_486 = (var_6 == var_485);
                                                                        var_482 = var_482 || var_486;
                                                                    }
                                                                    if (var_482) {
                                                                        // pnts = vec6(           <L 685>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 686>
                                                                        var_488 = wp::address(var_sensor_collision_in, var_0, var_173, var_475, var_487);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 687>
                                                                        var_490 = wp::address(var_sensor_collision_in, var_0, var_173, var_475, var_489);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 688>
                                                                        var_492 = wp::address(var_sensor_collision_in, var_0, var_173, var_475, var_491);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 689>
                                                                        var_494 = wp::address(var_sensor_collision_in, var_0, var_173, var_475, var_493);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 690>
                                                                        var_496 = wp::address(var_sensor_collision_in, var_0, var_173, var_475, var_495);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 691>
                                                                        var_498 = wp::address(var_sensor_collision_in, var_0, var_173, var_475, var_497);
                                                                        var_500 = wp::load(var_488);
                                                                        var_501 = wp::load(var_490);
                                                                        var_502 = wp::load(var_492);
                                                                        var_503 = wp::load(var_494);
                                                                        var_504 = wp::load(var_496);
                                                                        var_505 = wp::load(var_498);
                                                                        var_499 = wp::vec_t<6, wp::float32>({var_500, var_501, var_502, var_503, var_504, var_505});
                                                                    }
                                                                    var_506 = wp::where(var_482, var_499, var_473);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 694>
                                                                    var_507 = wp::address(var_geom_type, var_163);
                                                                    var_508 = wp::address(var_geom_type, var_166);
                                                                    var_510 = wp::load(var_507);
                                                                    var_511 = wp::load(var_508);
                                                                    var_509 = (var_510 > var_511);
                                                                    if (var_509) {
                                                                        // flip = True            <L 695>
                                                                    }
                                                                    if (!var_509) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 696>
                                                                        var_513 = wp::address(var_geom_type, var_163);
                                                                        var_514 = wp::address(var_geom_type, var_166);
                                                                        var_516 = wp::load(var_513);
                                                                        var_517 = wp::load(var_514);
                                                                        var_515 = (var_516 == var_517);
                                                                        if (var_515) {
                                                                            // flip = geomid1 > geomid2       <L 697>
                                                                            var_518 = (var_163 > var_166);
                                                                        }
                                                                        if (!var_515) {
                                                                            // flip = False       <L 699>
                                                                        }
                                                                        var_520 = wp::where(var_515, var_518, var_519);
                                                                    }
                                                                    var_521 = wp::where(var_509, var_512, var_520);
                                                                }
                                                                var_522 = wp::where(var_480, var_481, var_472);
                                                                var_523 = wp::where(var_480, var_506, var_473);
                                                                var_524 = wp::where(var_480, var_521, var_474);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 679>
                                                                var_527 = wp::address(var_sensor_collision_in, var_0, var_173, var_525, var_526);
                                                                var_529 = wp::load(var_527);
                                                                var_528 = wp::copy(var_529);
                                                                // if dist_new < dist:            <L 681>
                                                                var_530 = (var_528 < var_522);
                                                                if (var_530) {
                                                                    // dist = dist_new            <L 682>
                                                                    var_531 = wp::copy(var_528);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 684>
                                                                    var_534 = (var_6 == var_533);
                                                                    var_532 = var_534;
                                                                    if (!var_532) {
                                                                        var_536 = (var_6 == var_535);
                                                                        var_532 = var_532 || var_536;
                                                                    }
                                                                    if (var_532) {
                                                                        // pnts = vec6(           <L 685>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 686>
                                                                        var_538 = wp::address(var_sensor_collision_in, var_0, var_173, var_525, var_537);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 687>
                                                                        var_540 = wp::address(var_sensor_collision_in, var_0, var_173, var_525, var_539);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 688>
                                                                        var_542 = wp::address(var_sensor_collision_in, var_0, var_173, var_525, var_541);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 689>
                                                                        var_544 = wp::address(var_sensor_collision_in, var_0, var_173, var_525, var_543);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 690>
                                                                        var_546 = wp::address(var_sensor_collision_in, var_0, var_173, var_525, var_545);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 691>
                                                                        var_548 = wp::address(var_sensor_collision_in, var_0, var_173, var_525, var_547);
                                                                        var_550 = wp::load(var_538);
                                                                        var_551 = wp::load(var_540);
                                                                        var_552 = wp::load(var_542);
                                                                        var_553 = wp::load(var_544);
                                                                        var_554 = wp::load(var_546);
                                                                        var_555 = wp::load(var_548);
                                                                        var_549 = wp::vec_t<6, wp::float32>({var_550, var_551, var_552, var_553, var_554, var_555});
                                                                    }
                                                                    var_556 = wp::where(var_532, var_549, var_523);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 694>
                                                                    var_557 = wp::address(var_geom_type, var_163);
                                                                    var_558 = wp::address(var_geom_type, var_166);
                                                                    var_560 = wp::load(var_557);
                                                                    var_561 = wp::load(var_558);
                                                                    var_559 = (var_560 > var_561);
                                                                    if (var_559) {
                                                                        // flip = True            <L 695>
                                                                    }
                                                                    if (!var_559) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 696>
                                                                        var_563 = wp::address(var_geom_type, var_163);
                                                                        var_564 = wp::address(var_geom_type, var_166);
                                                                        var_566 = wp::load(var_563);
                                                                        var_567 = wp::load(var_564);
                                                                        var_565 = (var_566 == var_567);
                                                                        if (var_565) {
                                                                            // flip = geomid1 > geomid2       <L 697>
                                                                            var_568 = (var_163 > var_166);
                                                                        }
                                                                        if (!var_565) {
                                                                            // flip = False       <L 699>
                                                                        }
                                                                        var_570 = wp::where(var_565, var_568, var_569);
                                                                    }
                                                                    var_571 = wp::where(var_559, var_562, var_570);
                                                                }
                                                                var_572 = wp::where(var_530, var_531, var_522);
                                                                var_573 = wp::where(var_530, var_556, var_523);
                                                                var_574 = wp::where(var_530, var_571, var_524);
                                                                wp::assign(var_122, var_572);
                                                                wp::assign(var_130, var_573);
                                                                wp::assign(var_132, var_574);
                                                                goto start_for_2;
                                                            end_for_2:;
                                                            goto start_for_0;
                                                        end_for_0:;
                                                        // if sensortype == int(SensorType.GEOMDIST.value):       <L 700>
                                                        var_577 = wp::int(var_576);
                                                        var_578 = (var_6 == var_577);
                                                        if (var_578) {
                                                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, dist, out)       <L 701>
                                                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_122, var_11);
                                                        }
                                                        if (!var_578) {
                                                            // elif sensortype == int(SensorType.GEOMNORMAL.value):       <L 702>
                                                            var_581 = wp::int(var_580);
                                                            var_582 = (var_6 == var_581);
                                                            if (var_582) {
                                                                // if dist <= sensor_cutoff[sensorid]:       <L 703>
                                                                var_583 = wp::address(var_sensor_cutoff, var_3);
                                                                var_585 = wp::load(var_583);
                                                                var_584 = (var_122 <= var_585);
                                                                if (var_584) {
                                                                    // normal = wp.normalize(wp.vec3(pnts[3] - pnts[0], pnts[4] - pnts[1], pnts[5] - pnts[2]))       <L 704>
                                                                    var_587 = wp::extract(var_130, var_586);
                                                                    var_589 = wp::extract(var_130, var_588);
                                                                    var_590 = wp::sub(var_587, var_589);
                                                                    var_592 = wp::extract(var_130, var_591);
                                                                    var_594 = wp::extract(var_130, var_593);
                                                                    var_595 = wp::sub(var_592, var_594);
                                                                    var_597 = wp::extract(var_130, var_596);
                                                                    var_599 = wp::extract(var_130, var_598);
                                                                    var_600 = wp::sub(var_597, var_599);
                                                                    var_601 = wp::vec_t<3, wp::float32>(var_590, var_595, var_600);
                                                                    var_602 = wp::normalize(var_601);
                                                                    // if flip:                   <L 705>
                                                                    if (var_132) {
                                                                        // normal *= -1.0         <L 706>
                                                                        var_604 = wp::mul(var_602, var_603);
                                                                    }
                                                                    var_605 = wp::where(var_132, var_604, var_602);
                                                                }
                                                                if (!var_584) {
                                                                    // normal = wp.vec3(0.0, 0.0, 0.0)       <L 708>
                                                                    var_609 = wp::vec_t<3, wp::float32>(var_606, var_607, var_608);
                                                                }
                                                                var_610 = wp::where(var_584, var_605, var_609);
                                                                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, normal, out)       <L 709>
                                                                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_611, var_610, var_11);
                                                            }
                                                            if (!var_582) {
                                                                // elif sensortype == int(SensorType.GEOMFROMTO.value):       <L 710>
                                                                var_614 = wp::int(var_613);
                                                                var_615 = (var_6 == var_614);
                                                                if (var_615) {
                                                                    // if dist <= sensor_cutoff[sensorid]:       <L 711>
                                                                    var_616 = wp::address(var_sensor_cutoff, var_3);
                                                                    var_618 = wp::load(var_616);
                                                                    var_617 = (var_122 <= var_618);
                                                                    if (var_617) {
                                                                        // if flip:               <L 712>
                                                                        if (var_132) {
                                                                            // fromto = vec6(pnts[3], pnts[4], pnts[5], pnts[0], pnts[1], pnts[2])       <L 713>
                                                                            var_620 = wp::extract(var_130, var_619);
                                                                            var_622 = wp::extract(var_130, var_621);
                                                                            var_624 = wp::extract(var_130, var_623);
                                                                            var_626 = wp::extract(var_130, var_625);
                                                                            var_628 = wp::extract(var_130, var_627);
                                                                            var_630 = wp::extract(var_130, var_629);
                                                                            var_631 = wp::vec_t<6, wp::float32>({var_620, var_622, var_624, var_626, var_628, var_630});
                                                                        }
                                                                        if (!var_132) {
                                                                            // fromto = pnts       <L 715>
                                                                            var_632 = wp::copy(var_130);
                                                                        }
                                                                        var_633 = wp::where(var_132, var_631, var_632);
                                                                    }
                                                                    if (!var_617) {
                                                                        // fromto = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 717>
                                                                        var_640 = wp::vec_t<6, wp::float32>({var_634, var_635, var_636, var_637, var_638, var_639});
                                                                    }
                                                                    var_641 = wp::where(var_617, var_633, var_640);
                                                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 6, fromto, out)       <L 718>
                                                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_642, var_641, var_11);
                                                                }
                                                            }
                                                        }
                                                    }
                                                    if (!var_102) {
                                                        // elif sensortype == SensorType.INSIDESITE:       <L 719>
                                                        var_644 = (var_6 == var_643);
                                                        if (var_644) {
                                                            // objtype = sensor_objtype[sensorid]       <L 720>
                                                            var_645 = wp::address(var_sensor_objtype, var_3);
                                                            var_647 = wp::load(var_645);
                                                            var_646 = wp::copy(var_647);
                                                            // if objtype == ObjType.XBODY:       <L 721>
                                                            var_649 = (var_646 == var_648);
                                                            if (var_649) {
                                                                // xpos = xpos_in[worldid, objid]       <L 722>
                                                                var_650 = wp::address(var_xpos_in, var_0, var_9);
                                                                var_652 = wp::load(var_650);
                                                                var_651 = wp::copy(var_652);
                                                            }
                                                            if (!var_649) {
                                                                // elif objtype == ObjType.BODY:       <L 723>
                                                                var_654 = (var_646 == var_653);
                                                                if (var_654) {
                                                                    // if objid > 0:              <L 726>
                                                                    var_656 = (var_9 > var_655);
                                                                    if (var_656) {
                                                                        // if (                   <L 727>
                                                                        // body_mass[worldid % body_mass.shape[0], objid] < MJ_MINVAL       <L 728>
                                                                        var_658 = &(var_body_mass.shape);
                                                                        var_661 = wp::load(var_658);
                                                                        var_660 = wp::extract(var_661, var_659);
                                                                        var_662 = wp::mod(var_0, var_660);
                                                                        var_663 = wp::address(var_body_mass, var_662, var_9);
                                                                        var_666 = wp::load(var_663);
                                                                        var_665 = (var_666 < var_664);
                                                                        var_657 = var_665;
                                                                        if (var_657) {
                                                                            // and body_subtreemass[worldid % body_subtreemass.shape[0], objid] >= MJ_MINVAL       <L 729>
                                                                            var_667 = &(var_body_subtreemass.shape);
                                                                            var_670 = wp::load(var_667);
                                                                            var_669 = wp::extract(var_670, var_668);
                                                                            var_671 = wp::mod(var_0, var_669);
                                                                            var_672 = wp::address(var_body_subtreemass, var_671, var_9);
                                                                            var_674 = wp::load(var_672);
                                                                            var_673 = (var_674 >= var_664);
                                                                            var_657 = var_657 && var_673;
                                                                        }
                                                                        if (var_657) {
                                                                            // xpos = subtree_com_in[worldid, objid]       <L 731>
                                                                            var_675 = wp::address(var_subtree_com_in, var_0, var_9);
                                                                            var_677 = wp::load(var_675);
                                                                            var_676 = wp::copy(var_677);
                                                                        }
                                                                        if (!var_657) {
                                                                            // xpos = xipos_in[worldid, objid]       <L 733>
                                                                            var_678 = wp::address(var_xipos_in, var_0, var_9);
                                                                            var_680 = wp::load(var_678);
                                                                            var_679 = wp::copy(var_680);
                                                                        }
                                                                        var_681 = wp::where(var_657, var_676, var_679);
                                                                    }
                                                                    if (!var_656) {
                                                                        // xpos = xipos_in[worldid, objid]       <L 735>
                                                                        var_682 = wp::address(var_xipos_in, var_0, var_9);
                                                                        var_684 = wp::load(var_682);
                                                                        var_683 = wp::copy(var_684);
                                                                    }
                                                                    var_685 = wp::where(var_656, var_681, var_683);
                                                                }
                                                                if (!var_654) {
                                                                    // elif objtype == ObjType.GEOM:       <L 736>
                                                                    var_687 = (var_646 == var_686);
                                                                    if (var_687) {
                                                                        // xpos = geom_xpos_in[worldid, objid]       <L 737>
                                                                        var_688 = wp::address(var_geom_xpos_in, var_0, var_9);
                                                                        var_690 = wp::load(var_688);
                                                                        var_689 = wp::copy(var_690);
                                                                    }
                                                                    if (!var_687) {
                                                                        // elif objtype == ObjType.SITE:       <L 738>
                                                                        var_692 = (var_646 == var_691);
                                                                        if (var_692) {
                                                                            // xpos = site_xpos_in[worldid, objid]       <L 739>
                                                                            var_693 = wp::address(var_site_xpos_in, var_0, var_9);
                                                                            var_695 = wp::load(var_693);
                                                                            var_694 = wp::copy(var_695);
                                                                        }
                                                                        if (!var_692) {
                                                                            // elif objtype == ObjType.CAMERA:       <L 740>
                                                                            var_697 = (var_646 == var_696);
                                                                            if (var_697) {
                                                                                // xpos = cam_xpos_in[worldid, objid]       <L 741>
                                                                                var_698 = wp::address(var_cam_xpos_in, var_0, var_9);
                                                                                var_700 = wp::load(var_698);
                                                                                var_699 = wp::copy(var_700);
                                                                            }
                                                                            if (!var_697) {
                                                                                // return  # should not occur       <L 743>
                                                                                return;
                                                                            }
                                                                        }
                                                                        var_701 = wp::where(var_692, var_694, var_699);
                                                                    }
                                                                    var_702 = wp::where(var_687, var_689, var_701);
                                                                }
                                                                var_703 = wp::where(var_654, var_685, var_702);
                                                            }
                                                            var_704 = wp::where(var_649, var_651, var_703);
                                                            // refid = sensor_refid[sensorid]       <L 744>
                                                            var_705 = wp::address(var_sensor_refid, var_3);
                                                            var_707 = wp::load(var_705);
                                                            var_706 = wp::copy(var_707);
                                                            // val_bool = inside_geom(site_xpos_in[worldid, refid], site_xmat_in[worldid, refid], site_size[refid], site_type[refid], xpos)       <L 745>
                                                            var_708 = wp::address(var_site_xpos_in, var_0, var_706);
                                                            var_709 = wp::address(var_site_xmat_in, var_0, var_706);
                                                            var_710 = wp::address(var_site_size, var_706);
                                                            var_711 = wp::address(var_site_type, var_706);
                                                            var_713 = wp::load(var_708);
                                                            var_714 = wp::load(var_709);
                                                            var_715 = wp::load(var_710);
                                                            var_716 = wp::load(var_711);
                                                            var_712 = inside_geom_0(var_713, var_714, var_715, var_716, var_704);
                                                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, float(val_bool), out)       <L 746>
                                                            var_717 = wp::float(var_712);
                                                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_717, var_11);
                                                        }
                                                        if (!var_644) {
                                                            // elif sensortype == SensorType.E_POTENTIAL:       <L 747>
                                                            var_719 = (var_6 == var_718);
                                                            if (var_719) {
                                                                // val = energy_in[worldid][0]       <L 748>
                                                                var_720 = wp::address(var_energy_in, var_0);
                                                                var_723 = wp::load(var_720);
                                                                var_722 = wp::extract(var_723, var_721);
                                                                // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 749>
                                                                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_722, var_11);
                                                            }
                                                            if (!var_719) {
                                                                // elif sensortype == SensorType.E_KINETIC:       <L 750>
                                                                var_725 = (var_6 == var_724);
                                                                if (var_725) {
                                                                    // val = energy_in[worldid][1]       <L 751>
                                                                    var_726 = wp::address(var_energy_in, var_0);
                                                                    var_729 = wp::load(var_726);
                                                                    var_728 = wp::extract(var_729, var_727);
                                                                    // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 752>
                                                                    _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_728, var_11);
                                                                }
                                                                if (!var_725) {
                                                                    // elif sensortype == SensorType.CLOCK:       <L 753>
                                                                    var_731 = (var_6 == var_730);
                                                                    if (var_731) {
                                                                        // val = _clock(time_in, worldid)       <L 754>
                                                                        var_732 = _clock_0(var_time_in, var_0);
                                                                        // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 755>
                                                                        _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_732, var_11);
                                                                    }
                                                                }
                                                                var_733 = wp::where(var_725, var_728, var_732);
                                                            }
                                                            var_734 = wp::where(var_719, var_722, var_733);
                                                        }
                                                    }
                                                    var_735 = wp::where(var_102, var_113, var_9);
                                                    var_736 = wp::where(var_102, var_110, var_646);
                                                    var_737 = wp::where(var_102, var_119, var_706);
                                                }
                                                var_738 = wp::where(var_99, var_9, var_735);
                                            }
                                            var_739 = wp::where(var_86, var_9, var_738);
                                            var_740 = wp::where(var_86, var_88, var_736);
                                            var_741 = wp::where(var_86, var_91, var_737);
                                            var_742 = wp::where(var_86, var_94, var_116);
                                        }
                                        var_743 = wp::where(var_56, var_9, var_739);
                                        var_744 = wp::where(var_56, var_64, var_740);
                                        var_745 = wp::where(var_56, var_67, var_741);
                                        var_746 = wp::where(var_56, var_70, var_742);
                                        var_747 = wp::where(var_56, var_83, var_100);
                                    }
                                    var_748 = wp::where(var_44, var_9, var_743);
                                    var_749 = wp::where(var_44, var_46, var_744);
                                    var_750 = wp::where(var_44, var_49, var_745);
                                    var_751 = wp::where(var_44, var_52, var_746);
                                    var_752 = wp::where(var_44, var_54, var_747);
                                }
                                var_753 = wp::where(var_40, var_9, var_748);
                                var_754 = wp::where(var_40, var_41, var_96);
                            }
                            var_755 = wp::where(var_37, var_9, var_753);
                            var_756 = wp::where(var_37, var_38, var_734);
                        }
                        var_757 = wp::where(var_34, var_9, var_755);
                        var_758 = wp::where(var_34, var_35, var_756);
                    }
                    var_759 = wp::where(var_31, var_9, var_757);
                    var_760 = wp::where(var_31, var_32, var_758);
                }
                var_761 = wp::where(var_24, var_9, var_759);
                var_762 = wp::where(var_24, var_28, var_760);
            }
            var_763 = wp::where(var_17, var_9, var_761);
            var_764 = wp::where(var_17, var_19, var_750);
        }
        var_765 = wp::where(var_13, var_9, var_763);
        var_766 = wp::where(var_13, var_14, var_752);
}

