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



extern "C" __global__ void validate_and_correct_inertia_kernel_7044ee8d_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inertia,
    wp::array_t<wp::float32> var_body_inv_mass,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia,
    bool var_balance_inertia,
    wp::float32 var_bound_mass,
    wp::float32 var_bound_inertia,
    wp::array_t<wp::int32> var_correction_count)
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
        wp::float32* var_1;
        wp::float32 var_2;
        wp::float32 var_3;
        wp::mat_t<3, 3, wp::float32>* var_4;
        wp::mat_t<3, 3, wp::float32> var_5;
        wp::mat_t<3, 3, wp::float32> var_6;
        wp::mat_t<3, 3, wp::float32> var_7;
        const bool var_8 = false;
        bool var_9;
        bool var_10;
        bool var_11;
        const wp::int32 var_12 = 0;
        const wp::int32 var_13 = 0;
        wp::float32 var_14;
        bool var_15;
        bool var_16;
        const wp::int32 var_17 = 0;
        const wp::int32 var_18 = 1;
        wp::float32 var_19;
        bool var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        const wp::int32 var_23 = 2;
        wp::float32 var_24;
        bool var_25;
        bool var_26;
        const wp::int32 var_27 = 1;
        const wp::int32 var_28 = 0;
        wp::float32 var_29;
        bool var_30;
        bool var_31;
        const wp::int32 var_32 = 1;
        const wp::int32 var_33 = 1;
        wp::float32 var_34;
        bool var_35;
        bool var_36;
        const wp::int32 var_37 = 1;
        const wp::int32 var_38 = 2;
        wp::float32 var_39;
        bool var_40;
        bool var_41;
        const wp::int32 var_42 = 2;
        const wp::int32 var_43 = 0;
        wp::float32 var_44;
        bool var_45;
        bool var_46;
        const wp::int32 var_47 = 2;
        const wp::int32 var_48 = 1;
        wp::float32 var_49;
        bool var_50;
        bool var_51;
        const wp::int32 var_52 = 2;
        const wp::int32 var_53 = 2;
        wp::float32 var_54;
        bool var_55;
        bool var_56;
        const wp::float32 var_57 = 0.0;
        const wp::float32 var_58 = 0.0;
        const wp::float32 var_59 = 0.0;
        const wp::float32 var_60 = 0.0;
        const wp::float32 var_61 = 0.0;
        const wp::float32 var_62 = 0.0;
        const wp::float32 var_63 = 0.0;
        const wp::float32 var_64 = 0.0;
        const wp::float32 var_65 = 0.0;
        const wp::float32 var_66 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_67;
        const bool var_68 = true;
        wp::float32 var_69;
        wp::mat_t<3, 3, wp::float32> var_70;
        bool var_71;
        const wp::float32 var_72 = 0.0;
        bool var_73;
        const wp::float32 var_74 = 0.0;
        const bool var_75 = true;
        wp::float32 var_76;
        bool var_77;
        bool var_78;
        const wp::float32 var_79 = 0.0;
        bool var_80;
        bool var_81;
        const wp::float32 var_82 = 0.0;
        bool var_83;
        wp::float32 var_84;
        const bool var_85 = true;
        wp::float32 var_86;
        bool var_87;
        const wp::float32 var_88 = 0.0;
        bool var_89;
        bool var_90;
        wp::float32 var_91;
        const wp::float32 var_92 = 0.0;
        bool var_93;
        const wp::float32 var_94 = 0.0;
        const wp::float32 var_95 = 0.0;
        const wp::float32 var_96 = 0.0;
        const wp::float32 var_97 = 0.0;
        const wp::float32 var_98 = 0.0;
        const wp::float32 var_99 = 0.0;
        const wp::float32 var_100 = 0.0;
        const wp::float32 var_101 = 0.0;
        const wp::float32 var_102 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_103;
        const wp::int32 var_104 = 0;
        const wp::int32 var_105 = 1;
        wp::float32 var_106;
        const wp::int32 var_107 = 1;
        const wp::int32 var_108 = 0;
        wp::float32 var_109;
        wp::float32 var_110;
        const wp::float32 var_111 = 0.5;
        wp::float32 var_112;
        const wp::int32 var_113 = 0;
        const wp::int32 var_114 = 2;
        wp::float32 var_115;
        const wp::int32 var_116 = 2;
        const wp::int32 var_117 = 0;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::float32 var_120 = 0.5;
        wp::float32 var_121;
        const wp::int32 var_122 = 1;
        const wp::int32 var_123 = 2;
        wp::float32 var_124;
        const wp::int32 var_125 = 2;
        const wp::int32 var_126 = 1;
        wp::float32 var_127;
        wp::float32 var_128;
        const wp::float32 var_129 = 0.5;
        wp::float32 var_130;
        const wp::int32 var_131 = 0;
        const wp::int32 var_132 = 0;
        wp::float32 var_133;
        const wp::int32 var_134 = 1;
        const wp::int32 var_135 = 1;
        wp::float32 var_136;
        const wp::int32 var_137 = 2;
        const wp::int32 var_138 = 2;
        wp::float32 var_139;
        wp::mat_t<3, 3, wp::float32> var_140;
        const wp::float32 var_141 = 1e-08;
        const wp::float32 var_142 = 1e-05;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        bool var_152;
        const wp::int32 var_153 = 0;
        const wp::int32 var_154 = 1;
        wp::float32 var_155;
        wp::float32 var_156;
        wp::float32 var_157;
        bool var_158;
        const wp::int32 var_159 = 1;
        const wp::int32 var_160 = 0;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        bool var_164;
        const wp::int32 var_165 = 0;
        const wp::int32 var_166 = 2;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        bool var_170;
        const wp::int32 var_171 = 2;
        const wp::int32 var_172 = 0;
        wp::float32 var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        bool var_176;
        const wp::int32 var_177 = 1;
        const wp::int32 var_178 = 2;
        wp::float32 var_179;
        wp::float32 var_180;
        wp::float32 var_181;
        bool var_182;
        const wp::int32 var_183 = 2;
        const wp::int32 var_184 = 1;
        wp::float32 var_185;
        wp::float32 var_186;
        wp::float32 var_187;
        bool var_188;
        const bool var_189 = true;
        bool var_190;
        wp::mat_t<3, 3, wp::float32> var_191;
        wp::mat_t<3, 3, wp::float32> var_192;
        wp::vec_t<3, wp::float32> var_193;
        const wp::int32 var_194 = 0;
        wp::float32 var_195;
        const wp::int32 var_196 = 1;
        wp::float32 var_197;
        const wp::int32 var_198 = 2;
        wp::float32 var_199;
        bool var_200;
        wp::float32 var_201;
        wp::float32 var_202;
        bool var_203;
        bool var_204;
        wp::float32 var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        wp::float32 var_209;
        const wp::float32 var_210 = 1e-06;
        wp::float32 var_211;
        const wp::float32 var_212 = 1e-10;
        wp::float32 var_213;
        bool var_214;
        wp::float32 var_215;
        const wp::float32 var_216 = 1e-06;
        wp::float32 var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        const wp::float32 var_221 = 0.0;
        const wp::float32 var_222 = 0.0;
        const wp::float32 var_223 = 0.0;
        const wp::float32 var_224 = 0.0;
        const wp::float32 var_225 = 0.0;
        const wp::float32 var_226 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_227;
        wp::mat_t<3, 3, wp::float32> var_228;
        const bool var_229 = true;
        wp::mat_t<3, 3, wp::float32> var_230;
        bool var_231;
        wp::float32 var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        bool var_235;
        const wp::float32 var_236 = 0.0;
        bool var_237;
        bool var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        const wp::float32 var_243 = 0.0;
        const wp::float32 var_244 = 0.0;
        const wp::float32 var_245 = 0.0;
        const wp::float32 var_246 = 0.0;
        const wp::float32 var_247 = 0.0;
        const wp::float32 var_248 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_249;
        wp::mat_t<3, 3, wp::float32> var_250;
        const bool var_251 = true;
        wp::mat_t<3, 3, wp::float32> var_252;
        bool var_253;
        wp::float32 var_254;
        wp::float32 var_255;
        wp::float32 var_256;
        wp::float32 var_257;
        const wp::float32 var_258 = 1.1920929e-07;
        wp::float32 var_259;
        const wp::float32 var_260 = 1e-10;
        wp::float32 var_261;
        bool var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        bool var_265;
        wp::float32 var_266;
        wp::float32 var_267;
        const wp::float32 var_268 = 1e-06;
        wp::float32 var_269;
        const wp::float32 var_270 = 0.0;
        const wp::float32 var_271 = 0.0;
        const wp::float32 var_272 = 0.0;
        const wp::float32 var_273 = 0.0;
        const wp::float32 var_274 = 0.0;
        const wp::float32 var_275 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_276;
        wp::mat_t<3, 3, wp::float32> var_277;
        const bool var_278 = true;
        wp::mat_t<3, 3, wp::float32> var_279;
        bool var_280;
        wp::float32 var_281;
        wp::mat_t<3, 3, wp::float32> var_282;
        bool var_283;
        wp::mat_t<3, 3, wp::float32> var_284;
        const wp::float32 var_285 = 0.0;
        bool var_286;
        const wp::float32 var_287 = 1.0;
        wp::float32 var_288;
        const wp::float32 var_289 = 0.0;
        const wp::float32 var_290 = 0.0;
        bool var_291;
        wp::mat_t<3, 3, wp::float32> var_292;
        const wp::float32 var_293 = 0.0;
        const wp::float32 var_294 = 0.0;
        const wp::float32 var_295 = 0.0;
        const wp::float32 var_296 = 0.0;
        const wp::float32 var_297 = 0.0;
        const wp::float32 var_298 = 0.0;
        const wp::float32 var_299 = 0.0;
        const wp::float32 var_300 = 0.0;
        const wp::float32 var_301 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_302;
        const wp::int32 var_303 = 0;
        const wp::int32 var_304 = 1;
        wp::int32 var_305;
        //---------
        // forward
        // def validate_and_correct_inertia_kernel(                                               <L 965>
        // tid = wp.tid()                                                                         <L 981>
        var_0 = builtin_tid1d();
        // mass = body_mass[tid]                                                                  <L 983>
        var_1 = wp::address(var_body_mass, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // inertia = body_inertia[tid]                                                            <L 984>
        var_4 = wp::address(var_body_inertia, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // original_inertia = inertia                                                             <L 985>
        var_7 = wp::copy(var_5);
        // was_corrected = False                                                                  <L 986>
        // if (                                                                                   <L 989>
        // not wp.isfinite(mass)                                                                  <L 990>
        var_10 = wp::isfinite(var_2);
        var_11 = wp::unot(var_10);
        var_9 = var_11;
        if (!var_9) {
            // or not wp.isfinite(inertia[0, 0])                                                  <L 991>
            var_14 = wp::extract(var_5, var_12, var_13);
            var_15 = wp::isfinite(var_14);
            var_16 = wp::unot(var_15);
            var_9 = var_9 || var_16;
        }
        if (!var_9) {
            // or not wp.isfinite(inertia[0, 1])                                                  <L 992>
            var_19 = wp::extract(var_5, var_17, var_18);
            var_20 = wp::isfinite(var_19);
            var_21 = wp::unot(var_20);
            var_9 = var_9 || var_21;
        }
        if (!var_9) {
            // or not wp.isfinite(inertia[0, 2])                                                  <L 993>
            var_24 = wp::extract(var_5, var_22, var_23);
            var_25 = wp::isfinite(var_24);
            var_26 = wp::unot(var_25);
            var_9 = var_9 || var_26;
        }
        if (!var_9) {
            // or not wp.isfinite(inertia[1, 0])                                                  <L 994>
            var_29 = wp::extract(var_5, var_27, var_28);
            var_30 = wp::isfinite(var_29);
            var_31 = wp::unot(var_30);
            var_9 = var_9 || var_31;
        }
        if (!var_9) {
            // or not wp.isfinite(inertia[1, 1])                                                  <L 995>
            var_34 = wp::extract(var_5, var_32, var_33);
            var_35 = wp::isfinite(var_34);
            var_36 = wp::unot(var_35);
            var_9 = var_9 || var_36;
        }
        if (!var_9) {
            // or not wp.isfinite(inertia[1, 2])                                                  <L 996>
            var_39 = wp::extract(var_5, var_37, var_38);
            var_40 = wp::isfinite(var_39);
            var_41 = wp::unot(var_40);
            var_9 = var_9 || var_41;
        }
        if (!var_9) {
            // or not wp.isfinite(inertia[2, 0])                                                  <L 997>
            var_44 = wp::extract(var_5, var_42, var_43);
            var_45 = wp::isfinite(var_44);
            var_46 = wp::unot(var_45);
            var_9 = var_9 || var_46;
        }
        if (!var_9) {
            // or not wp.isfinite(inertia[2, 1])                                                  <L 998>
            var_49 = wp::extract(var_5, var_47, var_48);
            var_50 = wp::isfinite(var_49);
            var_51 = wp::unot(var_50);
            var_9 = var_9 || var_51;
        }
        if (!var_9) {
            // or not wp.isfinite(inertia[2, 2])                                                  <L 999>
            var_54 = wp::extract(var_5, var_52, var_53);
            var_55 = wp::isfinite(var_54);
            var_56 = wp::unot(var_55);
            var_9 = var_9 || var_56;
        }
        if (var_9) {
            // mass = 0.0                                                                         <L 1001>
            // inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                    <L 1002>
            var_67 = wp::mat_t<3, 3, wp::float32>(var_58, var_59, var_60, var_61, var_62, var_63, var_64, var_65, var_66);
            // was_corrected = True                                                               <L 1003>
        }
        var_69 = wp::where(var_9, var_57, var_2);
        var_70 = wp::where(var_9, var_67, var_5);
        var_71 = wp::where(var_9, var_68, var_8);
        // if mass < 0.0:                                                                         <L 1006>
        var_73 = (var_69 < var_72);
        if (var_73) {
            // mass = 0.0                                                                         <L 1007>
            // was_corrected = True                                                               <L 1008>
        }
        var_76 = wp::where(var_73, var_74, var_69);
        var_77 = wp::where(var_73, var_75, var_71);
        // if bound_mass > 0.0 and mass < bound_mass and mass > 0.0:                              <L 1011>
        var_80 = (var_bound_mass > var_79);
        var_78 = var_80;
        if (var_78) {
            var_81 = (var_76 < var_bound_mass);
            var_78 = var_78 && var_81;
        }
        if (var_78) {
            var_83 = (var_76 > var_82);
            var_78 = var_78 && var_83;
        }
        if (var_78) {
            // mass = bound_mass                                                                  <L 1012>
            var_84 = wp::copy(var_bound_mass);
            // was_corrected = True                                                               <L 1013>
        }
        var_86 = wp::where(var_78, var_84, var_76);
        var_87 = wp::where(var_78, var_85, var_77);
        // if mass == 0.0:                                                                        <L 1016>
        var_89 = (var_86 == var_88);
        if (var_89) {
            // was_corrected = was_corrected or (wp.ddot(inertia, inertia) > 0.0)                 <L 1017>
            var_90 = var_87;
            if (!var_90) {
                var_91 = wp::ddot(var_70, var_70);
                var_93 = (var_91 > var_92);
                var_90 = var_90 || var_93;
            }
            // inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                    <L 1018>
            var_103 = wp::mat_t<3, 3, wp::float32>(var_94, var_95, var_96, var_97, var_98, var_99, var_100, var_101, var_102);
        }
        if (!var_89) {
            // sym01 = (inertia[0, 1] + inertia[1, 0]) * 0.5                                      <L 1021>
            var_106 = wp::extract(var_70, var_104, var_105);
            var_109 = wp::extract(var_70, var_107, var_108);
            var_110 = wp::add(var_106, var_109);
            var_112 = wp::mul(var_110, var_111);
            // sym02 = (inertia[0, 2] + inertia[2, 0]) * 0.5                                      <L 1022>
            var_115 = wp::extract(var_70, var_113, var_114);
            var_118 = wp::extract(var_70, var_116, var_117);
            var_119 = wp::add(var_115, var_118);
            var_121 = wp::mul(var_119, var_120);
            // sym12 = (inertia[1, 2] + inertia[2, 1]) * 0.5                                      <L 1023>
            var_124 = wp::extract(var_70, var_122, var_123);
            var_127 = wp::extract(var_70, var_125, var_126);
            var_128 = wp::add(var_124, var_127);
            var_130 = wp::mul(var_128, var_129);
            // sym = wp.mat33(                                                                    <L 1024>
            // inertia[0, 0],                                                                     <L 1025>
            var_133 = wp::extract(var_70, var_131, var_132);
            // sym01,                                                                             <L 1026>
            // sym02,                                                                             <L 1027>
            // sym01,                                                                             <L 1028>
            // inertia[1, 1],                                                                     <L 1029>
            var_136 = wp::extract(var_70, var_134, var_135);
            // sym12,                                                                             <L 1030>
            // sym02,                                                                             <L 1031>
            // sym12,                                                                             <L 1032>
            // inertia[2, 2],                                                                     <L 1033>
            var_139 = wp::extract(var_70, var_137, var_138);
            var_140 = wp::mat_t<3, 3, wp::float32>(var_133, var_112, var_121, var_112, var_136, var_130, var_121, var_130, var_139);
            // tol01 = _INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * wp.abs(sym01)            <L 1036>
            var_143 = wp::abs(var_112);
            var_144 = wp::mul(var_142, var_143);
            var_145 = wp::add(var_141, var_144);
            // tol02 = _INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * wp.abs(sym02)            <L 1037>
            var_146 = wp::abs(var_121);
            var_147 = wp::mul(var_142, var_146);
            var_148 = wp::add(var_141, var_147);
            // tol12 = _INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * wp.abs(sym12)            <L 1038>
            var_149 = wp::abs(var_130);
            var_150 = wp::mul(var_142, var_149);
            var_151 = wp::add(var_141, var_150);
            // if (                                                                               <L 1039>
            // wp.abs(inertia[0, 1] - sym01) > tol01                                              <L 1040>
            var_155 = wp::extract(var_70, var_153, var_154);
            var_156 = wp::sub(var_155, var_112);
            var_157 = wp::abs(var_156);
            var_158 = (var_157 > var_145);
            var_152 = var_158;
            if (!var_152) {
                // or wp.abs(inertia[1, 0] - sym01) > tol01                                       <L 1041>
                var_161 = wp::extract(var_70, var_159, var_160);
                var_162 = wp::sub(var_161, var_112);
                var_163 = wp::abs(var_162);
                var_164 = (var_163 > var_145);
                var_152 = var_152 || var_164;
            }
            if (!var_152) {
                // or wp.abs(inertia[0, 2] - sym02) > tol02                                       <L 1042>
                var_167 = wp::extract(var_70, var_165, var_166);
                var_168 = wp::sub(var_167, var_121);
                var_169 = wp::abs(var_168);
                var_170 = (var_169 > var_148);
                var_152 = var_152 || var_170;
            }
            if (!var_152) {
                // or wp.abs(inertia[2, 0] - sym02) > tol02                                       <L 1043>
                var_173 = wp::extract(var_70, var_171, var_172);
                var_174 = wp::sub(var_173, var_121);
                var_175 = wp::abs(var_174);
                var_176 = (var_175 > var_148);
                var_152 = var_152 || var_176;
            }
            if (!var_152) {
                // or wp.abs(inertia[1, 2] - sym12) > tol12                                       <L 1044>
                var_179 = wp::extract(var_70, var_177, var_178);
                var_180 = wp::sub(var_179, var_130);
                var_181 = wp::abs(var_180);
                var_182 = (var_181 > var_151);
                var_152 = var_152 || var_182;
            }
            if (!var_152) {
                // or wp.abs(inertia[2, 1] - sym12) > tol12                                       <L 1045>
                var_185 = wp::extract(var_70, var_183, var_184);
                var_186 = wp::sub(var_185, var_130);
                var_187 = wp::abs(var_186);
                var_188 = (var_187 > var_151);
                var_152 = var_152 || var_188;
            }
            if (var_152) {
                // was_corrected = True                                                           <L 1047>
            }
            var_190 = wp::where(var_152, var_189, var_87);
            // inertia = sym                                                                      <L 1048>
            var_191 = wp::copy(var_140);
            // _eigvecs, eigvals = wp.eig3(inertia)                                               <L 1051>
            wp::eig3(var_191, var_192, var_193);
            // I1, I2, I3 = eigvals[0], eigvals[1], eigvals[2]                                    <L 1054>
            var_195 = wp::extract(var_193, var_194);
            var_197 = wp::extract(var_193, var_196);
            var_199 = wp::extract(var_193, var_198);
            // if I1 > I2:                                                                        <L 1055>
            var_200 = (var_195 > var_197);
            if (var_200) {
                // I1, I2 = I2, I1                                                                <L 1056>
            }
            var_201 = wp::where(var_200, var_197, var_195);
            var_202 = wp::where(var_200, var_195, var_197);
            // if I2 > I3:                                                                        <L 1057>
            var_203 = (var_202 > var_199);
            if (var_203) {
                // I2, I3 = I3, I2                                                                <L 1058>
                // if I1 > I2:                                                                    <L 1059>
                var_204 = (var_201 > var_199);
                if (var_204) {
                    // I1, I2 = I2, I1                                                            <L 1060>
                }
                var_205 = wp::where(var_204, var_199, var_201);
                var_206 = wp::where(var_204, var_201, var_199);
            }
            var_207 = wp::where(var_203, var_205, var_201);
            var_208 = wp::where(var_203, var_206, var_202);
            var_209 = wp::where(var_203, var_202, var_199);
            // eig_threshold = wp.max(1.0e-6 * I3, 1.0e-10)                                       <L 1064>
            var_211 = wp::mul(var_210, var_209);
            var_213 = wp::max(var_211, var_212);
            // if I1 < eig_threshold:                                                             <L 1065>
            var_214 = (var_207 < var_213);
            if (var_214) {
                // adjustment = eig_threshold - I1 + 1.0e-6                                       <L 1066>
                var_215 = wp::sub(var_213, var_207);
                var_217 = wp::add(var_215, var_216);
                // I1 += adjustment                                                               <L 1068>
                var_218 = wp::add(var_207, var_217);
                // I2 += adjustment                                                               <L 1069>
                var_219 = wp::add(var_208, var_217);
                // I3 += adjustment                                                               <L 1070>
                var_220 = wp::add(var_209, var_217);
                // inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)       <L 1071>
                var_227 = wp::mat_t<3, 3, wp::float32>(var_217, var_221, var_222, var_223, var_217, var_224, var_225, var_226, var_217);
                var_228 = wp::add(var_191, var_227);
                // was_corrected = True                                                           <L 1072>
            }
            var_230 = wp::where(var_214, var_228, var_191);
            var_231 = wp::where(var_214, var_229, var_190);
            var_232 = wp::where(var_214, var_218, var_207);
            var_233 = wp::where(var_214, var_219, var_208);
            var_234 = wp::where(var_214, var_220, var_209);
            // if bound_inertia > 0.0 and I1 < bound_inertia:                                     <L 1075>
            var_237 = (var_bound_inertia > var_236);
            var_235 = var_237;
            if (var_235) {
                var_238 = (var_232 < var_bound_inertia);
                var_235 = var_235 && var_238;
            }
            if (var_235) {
                // adjustment = bound_inertia - I1                                                <L 1076>
                var_239 = wp::sub(var_bound_inertia, var_232);
                // I1 += adjustment                                                               <L 1077>
                var_240 = wp::add(var_232, var_239);
                // I2 += adjustment                                                               <L 1078>
                var_241 = wp::add(var_233, var_239);
                // I3 += adjustment                                                               <L 1079>
                var_242 = wp::add(var_234, var_239);
                // inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)       <L 1080>
                var_249 = wp::mat_t<3, 3, wp::float32>(var_239, var_243, var_244, var_245, var_239, var_246, var_247, var_248, var_239);
                var_250 = wp::add(var_230, var_249);
                // was_corrected = True                                                           <L 1081>
            }
            var_252 = wp::where(var_235, var_250, var_230);
            var_253 = wp::where(var_235, var_251, var_231);
            var_254 = wp::where(var_235, var_240, var_232);
            var_255 = wp::where(var_235, var_241, var_233);
            var_256 = wp::where(var_235, var_242, var_234);
            var_257 = wp::where(var_235, var_239, var_217);
            // tri_tol = wp.max(1.1920929e-7 * I3, 1.0e-10)  # float32 eps * I3                   <L 1084>
            var_259 = wp::mul(var_258, var_256);
            var_261 = wp::max(var_259, var_260);
            // if balance_inertia and (I1 + I2 < I3 - tri_tol):                                   <L 1085>
            var_262 = var_balance_inertia;
            if (var_262) {
                var_263 = wp::add(var_254, var_255);
                var_264 = wp::sub(var_256, var_261);
                var_265 = (var_263 < var_264);
                var_262 = var_262 && var_265;
            }
            if (var_262) {
                // deficit = I3 - I1 - I2                                                         <L 1086>
                var_266 = wp::sub(var_256, var_254);
                var_267 = wp::sub(var_266, var_255);
                // adjustment = deficit + 1.0e-6                                                  <L 1087>
                var_269 = wp::add(var_267, var_268);
                // inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)       <L 1089>
                var_276 = wp::mat_t<3, 3, wp::float32>(var_269, var_270, var_271, var_272, var_269, var_273, var_274, var_275, var_269);
                var_277 = wp::add(var_252, var_276);
                // was_corrected = True                                                           <L 1090>
            }
            var_279 = wp::where(var_262, var_277, var_252);
            var_280 = wp::where(var_262, var_278, var_253);
            var_281 = wp::where(var_262, var_269, var_257);
        }
        var_282 = wp::where(var_89, var_103, var_279);
        var_283 = wp::where(var_89, var_90, var_280);
        // output_inertia = inertia if was_corrected else original_inertia                        <L 1092>
        if (var_283) {
        }
        if (!var_283) {
        }
        var_284 = wp::where(var_283, var_282, var_7);
        // body_mass[tid] = mass                                                                  <L 1095>
        wp::array_store(var_body_mass, var_0, var_86);
        // body_inertia[tid] = output_inertia                                                     <L 1096>
        wp::array_store(var_body_inertia, var_0, var_284);
        // if mass > 0.0:                                                                         <L 1099>
        var_286 = (var_86 > var_285);
        if (var_286) {
            // body_inv_mass[tid] = 1.0 / mass                                                    <L 1100>
            var_288 = wp::div(var_287, var_86);
            wp::array_store(var_body_inv_mass, var_0, var_288);
        }
        if (!var_286) {
            // body_inv_mass[tid] = 0.0                                                           <L 1102>
            wp::array_store(var_body_inv_mass, var_0, var_289);
        }
        // if mass > 0.0:                                                                         <L 1105>
        var_291 = (var_86 > var_290);
        if (var_291) {
            // body_inv_inertia[tid] = wp.inverse(output_inertia)                                 <L 1106>
            var_292 = wp::inverse(var_284);
            wp::array_store(var_body_inv_inertia, var_0, var_292);
        }
        if (!var_291) {
            // body_inv_inertia[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 1108>
            var_302 = wp::mat_t<3, 3, wp::float32>(var_293, var_294, var_295, var_296, var_297, var_298, var_299, var_300, var_301);
            wp::array_store(var_body_inv_inertia, var_0, var_302);
        }
        // if was_corrected:                                                                      <L 1110>
        if (var_283) {
            // wp.atomic_add(correction_count, 0, 1)                                              <L 1111>
            var_305 = wp::atomic_add(var_correction_count, var_303, var_304);
        }
    }
}

