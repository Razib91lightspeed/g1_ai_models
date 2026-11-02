#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 64
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



extern "C" __global__ void _small_cholesky_factorize_solve_block__locals__kernel_35b90ab6_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_qLD_block_adr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_block_dof,
    wp::array_t<wp::float32> var_y,
    wp::array_t<wp::float32> var_D_out,
    wp::array_t<wp::float32> var_x_out,
    wp::array_t<wp::float32> var_L_out)
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
        const wp::int32 var_5 = 6;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = -2;
        bool var_13;
        const wp::int32 var_14 = 0;
        const wp::float32 var_15 = 1.0;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        const wp::float32 var_27 = 1.0;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 2;
        const wp::float32 var_39 = 1.0;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::int32 var_49;
        const wp::int32 var_50 = 3;
        const wp::float32 var_51 = 1.0;
        wp::int32 var_52;
        wp::float32* var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::float32* var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::int32 var_61;
        const wp::int32 var_62 = 4;
        const wp::float32 var_63 = 1.0;
        wp::int32 var_64;
        wp::float32* var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::float32* var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::int32 var_73;
        const wp::int32 var_74 = 5;
        const wp::float32 var_75 = 1.0;
        wp::int32 var_76;
        wp::float32* var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::float32* var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::int32 var_85;
        const wp::int32 var_86 = 0;
        const wp::int32 var_87 = 1;
        wp::int32 var_88;
        wp::int32 var_89;
        const wp::int32 var_90 = 2;
        wp::int32 var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        wp::float32* var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::int32 var_97;
        wp::float32* var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::int32 var_104;
        const wp::float32 var_105 = 1.0;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::int32 var_108;
        const wp::int32 var_109 = 1;
        wp::int32 var_110;
        const wp::int32 var_111 = 6;
        wp::range_t var_112;
        wp::int32 var_113;
        const wp::int32 var_114 = 1;
        wp::int32 var_115;
        wp::int32 var_116;
        const wp::int32 var_117 = 2;
        wp::int32 var_118;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::float32* var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::int32 var_127;
        const wp::int32 var_128 = 1;
        const wp::int32 var_129 = 1;
        wp::int32 var_130;
        wp::int32 var_131;
        const wp::int32 var_132 = 2;
        wp::int32 var_133;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::float32* var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::int32 var_139;
        wp::float32* var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::int32 var_144;
        wp::int32 var_145;
        wp::int32 var_146;
        wp::float32* var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        wp::int32 var_152;
        wp::float32* var_153;
        wp::float32 var_154;
        wp::float32 var_155;
        wp::float32 var_156;
        wp::float32 var_157;
        wp::int32 var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        const wp::float32 var_161 = 1.0;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::int32 var_164;
        const wp::int32 var_165 = 1;
        wp::int32 var_166;
        const wp::int32 var_167 = 6;
        wp::range_t var_168;
        wp::int32 var_169;
        wp::int32 var_170;
        const wp::int32 var_171 = 1;
        wp::int32 var_172;
        wp::int32 var_173;
        const wp::int32 var_174 = 2;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::int32 var_177;
        wp::float32* var_178;
        wp::float32 var_179;
        wp::float32 var_180;
        const wp::int32 var_181 = 0;
        wp::int32 var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        wp::float32* var_185;
        wp::int32 var_186;
        wp::int32 var_187;
        wp::int32 var_188;
        wp::float32* var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::float32 var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::int32 var_195;
        wp::int32 var_196;
        wp::int32 var_197;
        const wp::int32 var_198 = 2;
        const wp::int32 var_199 = 1;
        wp::int32 var_200;
        wp::int32 var_201;
        const wp::int32 var_202 = 2;
        wp::int32 var_203;
        wp::int32 var_204;
        wp::int32 var_205;
        wp::float32* var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        wp::int32 var_209;
        wp::float32* var_210;
        wp::float32 var_211;
        wp::float32 var_212;
        const wp::int32 var_213 = 0;
        wp::int32 var_214;
        wp::int32 var_215;
        wp::int32 var_216;
        wp::float32* var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::int32 var_222;
        wp::float32* var_223;
        wp::float32 var_224;
        wp::float32 var_225;
        wp::float32 var_226;
        const wp::int32 var_227 = 1;
        wp::int32 var_228;
        wp::int32 var_229;
        wp::int32 var_230;
        wp::float32* var_231;
        wp::float32 var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::int32 var_236;
        wp::float32* var_237;
        wp::float32 var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        wp::int32 var_242;
        wp::int32 var_243;
        wp::int32 var_244;
        const wp::float32 var_245 = 1.0;
        wp::float32 var_246;
        wp::float32 var_247;
        wp::int32 var_248;
        const wp::int32 var_249 = 1;
        wp::int32 var_250;
        const wp::int32 var_251 = 6;
        wp::range_t var_252;
        wp::int32 var_253;
        wp::int32 var_254;
        const wp::int32 var_255 = 1;
        wp::int32 var_256;
        wp::int32 var_257;
        const wp::int32 var_258 = 2;
        wp::int32 var_259;
        wp::int32 var_260;
        wp::int32 var_261;
        wp::float32* var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        const wp::int32 var_265 = 0;
        wp::int32 var_266;
        wp::int32 var_267;
        wp::int32 var_268;
        wp::float32* var_269;
        wp::int32 var_270;
        wp::int32 var_271;
        wp::int32 var_272;
        wp::float32* var_273;
        wp::float32 var_274;
        wp::float32 var_275;
        wp::float32 var_276;
        wp::float32 var_277;
        const wp::int32 var_278 = 1;
        wp::int32 var_279;
        wp::int32 var_280;
        wp::int32 var_281;
        wp::float32* var_282;
        wp::int32 var_283;
        wp::int32 var_284;
        wp::int32 var_285;
        wp::float32* var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        wp::float32 var_289;
        wp::float32 var_290;
        wp::float32 var_291;
        wp::int32 var_292;
        wp::int32 var_293;
        wp::int32 var_294;
        const wp::int32 var_295 = 3;
        const wp::int32 var_296 = 1;
        wp::int32 var_297;
        wp::int32 var_298;
        const wp::int32 var_299 = 2;
        wp::int32 var_300;
        wp::int32 var_301;
        wp::int32 var_302;
        wp::float32* var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::int32 var_306;
        wp::float32* var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        const wp::int32 var_310 = 0;
        wp::int32 var_311;
        wp::int32 var_312;
        wp::int32 var_313;
        wp::float32* var_314;
        wp::float32 var_315;
        wp::float32 var_316;
        wp::float32 var_317;
        wp::float32 var_318;
        wp::int32 var_319;
        wp::float32* var_320;
        wp::float32 var_321;
        wp::float32 var_322;
        wp::float32 var_323;
        const wp::int32 var_324 = 1;
        wp::int32 var_325;
        wp::int32 var_326;
        wp::int32 var_327;
        wp::float32* var_328;
        wp::float32 var_329;
        wp::float32 var_330;
        wp::float32 var_331;
        wp::float32 var_332;
        wp::int32 var_333;
        wp::float32* var_334;
        wp::float32 var_335;
        wp::float32 var_336;
        wp::float32 var_337;
        const wp::int32 var_338 = 2;
        wp::int32 var_339;
        wp::int32 var_340;
        wp::int32 var_341;
        wp::float32* var_342;
        wp::float32 var_343;
        wp::float32 var_344;
        wp::float32 var_345;
        wp::float32 var_346;
        wp::int32 var_347;
        wp::float32* var_348;
        wp::float32 var_349;
        wp::float32 var_350;
        wp::float32 var_351;
        wp::float32 var_352;
        wp::int32 var_353;
        wp::int32 var_354;
        wp::int32 var_355;
        const wp::float32 var_356 = 1.0;
        wp::float32 var_357;
        wp::float32 var_358;
        wp::int32 var_359;
        const wp::int32 var_360 = 1;
        wp::int32 var_361;
        const wp::int32 var_362 = 6;
        wp::range_t var_363;
        wp::int32 var_364;
        wp::int32 var_365;
        const wp::int32 var_366 = 1;
        wp::int32 var_367;
        wp::int32 var_368;
        const wp::int32 var_369 = 2;
        wp::int32 var_370;
        wp::int32 var_371;
        wp::int32 var_372;
        wp::float32* var_373;
        wp::float32 var_374;
        wp::float32 var_375;
        const wp::int32 var_376 = 0;
        wp::int32 var_377;
        wp::int32 var_378;
        wp::int32 var_379;
        wp::float32* var_380;
        wp::int32 var_381;
        wp::int32 var_382;
        wp::int32 var_383;
        wp::float32* var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        wp::float32 var_387;
        wp::float32 var_388;
        const wp::int32 var_389 = 1;
        wp::int32 var_390;
        wp::int32 var_391;
        wp::int32 var_392;
        wp::float32* var_393;
        wp::int32 var_394;
        wp::int32 var_395;
        wp::int32 var_396;
        wp::float32* var_397;
        wp::float32 var_398;
        wp::float32 var_399;
        wp::float32 var_400;
        wp::float32 var_401;
        const wp::int32 var_402 = 2;
        wp::int32 var_403;
        wp::int32 var_404;
        wp::int32 var_405;
        wp::float32* var_406;
        wp::int32 var_407;
        wp::int32 var_408;
        wp::int32 var_409;
        wp::float32* var_410;
        wp::float32 var_411;
        wp::float32 var_412;
        wp::float32 var_413;
        wp::float32 var_414;
        wp::float32 var_415;
        wp::int32 var_416;
        wp::int32 var_417;
        wp::int32 var_418;
        const wp::int32 var_419 = 4;
        const wp::int32 var_420 = 1;
        wp::int32 var_421;
        wp::int32 var_422;
        const wp::int32 var_423 = 2;
        wp::int32 var_424;
        wp::int32 var_425;
        wp::int32 var_426;
        wp::float32* var_427;
        wp::float32 var_428;
        wp::float32 var_429;
        wp::int32 var_430;
        wp::float32* var_431;
        wp::float32 var_432;
        wp::float32 var_433;
        const wp::int32 var_434 = 0;
        wp::int32 var_435;
        wp::int32 var_436;
        wp::int32 var_437;
        wp::float32* var_438;
        wp::float32 var_439;
        wp::float32 var_440;
        wp::float32 var_441;
        wp::float32 var_442;
        wp::int32 var_443;
        wp::float32* var_444;
        wp::float32 var_445;
        wp::float32 var_446;
        wp::float32 var_447;
        const wp::int32 var_448 = 1;
        wp::int32 var_449;
        wp::int32 var_450;
        wp::int32 var_451;
        wp::float32* var_452;
        wp::float32 var_453;
        wp::float32 var_454;
        wp::float32 var_455;
        wp::float32 var_456;
        wp::int32 var_457;
        wp::float32* var_458;
        wp::float32 var_459;
        wp::float32 var_460;
        wp::float32 var_461;
        const wp::int32 var_462 = 2;
        wp::int32 var_463;
        wp::int32 var_464;
        wp::int32 var_465;
        wp::float32* var_466;
        wp::float32 var_467;
        wp::float32 var_468;
        wp::float32 var_469;
        wp::float32 var_470;
        wp::int32 var_471;
        wp::float32* var_472;
        wp::float32 var_473;
        wp::float32 var_474;
        wp::float32 var_475;
        const wp::int32 var_476 = 3;
        wp::int32 var_477;
        wp::int32 var_478;
        wp::int32 var_479;
        wp::float32* var_480;
        wp::float32 var_481;
        wp::float32 var_482;
        wp::float32 var_483;
        wp::float32 var_484;
        wp::int32 var_485;
        wp::float32* var_486;
        wp::float32 var_487;
        wp::float32 var_488;
        wp::float32 var_489;
        wp::float32 var_490;
        wp::int32 var_491;
        wp::int32 var_492;
        wp::int32 var_493;
        const wp::float32 var_494 = 1.0;
        wp::float32 var_495;
        wp::float32 var_496;
        wp::int32 var_497;
        const wp::int32 var_498 = 1;
        wp::int32 var_499;
        const wp::int32 var_500 = 6;
        wp::range_t var_501;
        wp::int32 var_502;
        wp::int32 var_503;
        const wp::int32 var_504 = 1;
        wp::int32 var_505;
        wp::int32 var_506;
        const wp::int32 var_507 = 2;
        wp::int32 var_508;
        wp::int32 var_509;
        wp::int32 var_510;
        wp::float32* var_511;
        wp::float32 var_512;
        wp::float32 var_513;
        const wp::int32 var_514 = 0;
        wp::int32 var_515;
        wp::int32 var_516;
        wp::int32 var_517;
        wp::float32* var_518;
        wp::int32 var_519;
        wp::int32 var_520;
        wp::int32 var_521;
        wp::float32* var_522;
        wp::float32 var_523;
        wp::float32 var_524;
        wp::float32 var_525;
        wp::float32 var_526;
        const wp::int32 var_527 = 1;
        wp::int32 var_528;
        wp::int32 var_529;
        wp::int32 var_530;
        wp::float32* var_531;
        wp::int32 var_532;
        wp::int32 var_533;
        wp::int32 var_534;
        wp::float32* var_535;
        wp::float32 var_536;
        wp::float32 var_537;
        wp::float32 var_538;
        wp::float32 var_539;
        const wp::int32 var_540 = 2;
        wp::int32 var_541;
        wp::int32 var_542;
        wp::int32 var_543;
        wp::float32* var_544;
        wp::int32 var_545;
        wp::int32 var_546;
        wp::int32 var_547;
        wp::float32* var_548;
        wp::float32 var_549;
        wp::float32 var_550;
        wp::float32 var_551;
        wp::float32 var_552;
        const wp::int32 var_553 = 3;
        wp::int32 var_554;
        wp::int32 var_555;
        wp::int32 var_556;
        wp::float32* var_557;
        wp::int32 var_558;
        wp::int32 var_559;
        wp::int32 var_560;
        wp::float32* var_561;
        wp::float32 var_562;
        wp::float32 var_563;
        wp::float32 var_564;
        wp::float32 var_565;
        wp::float32 var_566;
        wp::int32 var_567;
        wp::int32 var_568;
        wp::int32 var_569;
        const wp::int32 var_570 = 5;
        const wp::int32 var_571 = 1;
        wp::int32 var_572;
        wp::int32 var_573;
        const wp::int32 var_574 = 2;
        wp::int32 var_575;
        wp::int32 var_576;
        wp::int32 var_577;
        wp::float32* var_578;
        wp::float32 var_579;
        wp::float32 var_580;
        wp::int32 var_581;
        wp::float32* var_582;
        wp::float32 var_583;
        wp::float32 var_584;
        const wp::int32 var_585 = 0;
        wp::int32 var_586;
        wp::int32 var_587;
        wp::int32 var_588;
        wp::float32* var_589;
        wp::float32 var_590;
        wp::float32 var_591;
        wp::float32 var_592;
        wp::float32 var_593;
        wp::int32 var_594;
        wp::float32* var_595;
        wp::float32 var_596;
        wp::float32 var_597;
        wp::float32 var_598;
        const wp::int32 var_599 = 1;
        wp::int32 var_600;
        wp::int32 var_601;
        wp::int32 var_602;
        wp::float32* var_603;
        wp::float32 var_604;
        wp::float32 var_605;
        wp::float32 var_606;
        wp::float32 var_607;
        wp::int32 var_608;
        wp::float32* var_609;
        wp::float32 var_610;
        wp::float32 var_611;
        wp::float32 var_612;
        const wp::int32 var_613 = 2;
        wp::int32 var_614;
        wp::int32 var_615;
        wp::int32 var_616;
        wp::float32* var_617;
        wp::float32 var_618;
        wp::float32 var_619;
        wp::float32 var_620;
        wp::float32 var_621;
        wp::int32 var_622;
        wp::float32* var_623;
        wp::float32 var_624;
        wp::float32 var_625;
        wp::float32 var_626;
        const wp::int32 var_627 = 3;
        wp::int32 var_628;
        wp::int32 var_629;
        wp::int32 var_630;
        wp::float32* var_631;
        wp::float32 var_632;
        wp::float32 var_633;
        wp::float32 var_634;
        wp::float32 var_635;
        wp::int32 var_636;
        wp::float32* var_637;
        wp::float32 var_638;
        wp::float32 var_639;
        wp::float32 var_640;
        const wp::int32 var_641 = 4;
        wp::int32 var_642;
        wp::int32 var_643;
        wp::int32 var_644;
        wp::float32* var_645;
        wp::float32 var_646;
        wp::float32 var_647;
        wp::float32 var_648;
        wp::float32 var_649;
        wp::int32 var_650;
        wp::float32* var_651;
        wp::float32 var_652;
        wp::float32 var_653;
        wp::float32 var_654;
        wp::float32 var_655;
        wp::int32 var_656;
        wp::int32 var_657;
        wp::int32 var_658;
        const wp::float32 var_659 = 1.0;
        wp::float32 var_660;
        wp::float32 var_661;
        wp::int32 var_662;
        const wp::int32 var_663 = 1;
        wp::int32 var_664;
        const wp::int32 var_665 = 6;
        wp::range_t var_666;
        wp::int32 var_667;
        wp::int32 var_668;
        const wp::int32 var_669 = 1;
        wp::int32 var_670;
        wp::int32 var_671;
        const wp::int32 var_672 = 2;
        wp::int32 var_673;
        wp::int32 var_674;
        wp::int32 var_675;
        wp::float32* var_676;
        wp::float32 var_677;
        wp::float32 var_678;
        const wp::int32 var_679 = 0;
        wp::int32 var_680;
        wp::int32 var_681;
        wp::int32 var_682;
        wp::float32* var_683;
        wp::int32 var_684;
        wp::int32 var_685;
        wp::int32 var_686;
        wp::float32* var_687;
        wp::float32 var_688;
        wp::float32 var_689;
        wp::float32 var_690;
        wp::float32 var_691;
        const wp::int32 var_692 = 1;
        wp::int32 var_693;
        wp::int32 var_694;
        wp::int32 var_695;
        wp::float32* var_696;
        wp::int32 var_697;
        wp::int32 var_698;
        wp::int32 var_699;
        wp::float32* var_700;
        wp::float32 var_701;
        wp::float32 var_702;
        wp::float32 var_703;
        wp::float32 var_704;
        const wp::int32 var_705 = 2;
        wp::int32 var_706;
        wp::int32 var_707;
        wp::int32 var_708;
        wp::float32* var_709;
        wp::int32 var_710;
        wp::int32 var_711;
        wp::int32 var_712;
        wp::float32* var_713;
        wp::float32 var_714;
        wp::float32 var_715;
        wp::float32 var_716;
        wp::float32 var_717;
        const wp::int32 var_718 = 3;
        wp::int32 var_719;
        wp::int32 var_720;
        wp::int32 var_721;
        wp::float32* var_722;
        wp::int32 var_723;
        wp::int32 var_724;
        wp::int32 var_725;
        wp::float32* var_726;
        wp::float32 var_727;
        wp::float32 var_728;
        wp::float32 var_729;
        wp::float32 var_730;
        const wp::int32 var_731 = 4;
        wp::int32 var_732;
        wp::int32 var_733;
        wp::int32 var_734;
        wp::float32* var_735;
        wp::int32 var_736;
        wp::int32 var_737;
        wp::int32 var_738;
        wp::float32* var_739;
        wp::float32 var_740;
        wp::float32 var_741;
        wp::float32 var_742;
        wp::float32 var_743;
        wp::float32 var_744;
        wp::int32 var_745;
        wp::int32 var_746;
        wp::int32 var_747;
        const wp::int32 var_748 = 0;
        const wp::int32 var_749 = 1;
        wp::int32 var_750;
        wp::int32 var_751;
        wp::int32 var_752;
        wp::float32* var_753;
        wp::float32 var_754;
        wp::float32 var_755;
        const wp::int32 var_756 = 1;
        wp::int32 var_757;
        const wp::int32 var_758 = 6;
        wp::range_t var_759;
        wp::int32 var_760;
        wp::int32 var_761;
        wp::int32 var_762;
        wp::int32 var_763;
        wp::float32* var_764;
        wp::int32 var_765;
        wp::float32* var_766;
        wp::float32 var_767;
        wp::float32 var_768;
        wp::float32 var_769;
        wp::float32 var_770;
        wp::int32 var_771;
        wp::int32 var_772;
        wp::int32 var_773;
        wp::float32* var_774;
        wp::float32 var_775;
        wp::float32 var_776;
        wp::int32 var_777;
        const wp::int32 var_778 = 1;
        const wp::int32 var_779 = 1;
        wp::int32 var_780;
        wp::int32 var_781;
        wp::int32 var_782;
        wp::float32* var_783;
        wp::float32 var_784;
        wp::float32 var_785;
        const wp::int32 var_786 = 1;
        wp::int32 var_787;
        const wp::int32 var_788 = 6;
        wp::range_t var_789;
        wp::int32 var_790;
        wp::int32 var_791;
        wp::int32 var_792;
        wp::int32 var_793;
        wp::float32* var_794;
        wp::int32 var_795;
        wp::float32* var_796;
        wp::float32 var_797;
        wp::float32 var_798;
        wp::float32 var_799;
        wp::float32 var_800;
        wp::int32 var_801;
        wp::int32 var_802;
        wp::int32 var_803;
        wp::float32* var_804;
        wp::float32 var_805;
        wp::float32 var_806;
        wp::int32 var_807;
        const wp::int32 var_808 = 2;
        const wp::int32 var_809 = 1;
        wp::int32 var_810;
        wp::int32 var_811;
        wp::int32 var_812;
        wp::float32* var_813;
        wp::float32 var_814;
        wp::float32 var_815;
        const wp::int32 var_816 = 1;
        wp::int32 var_817;
        const wp::int32 var_818 = 6;
        wp::range_t var_819;
        wp::int32 var_820;
        wp::int32 var_821;
        wp::int32 var_822;
        wp::int32 var_823;
        wp::float32* var_824;
        wp::int32 var_825;
        wp::float32* var_826;
        wp::float32 var_827;
        wp::float32 var_828;
        wp::float32 var_829;
        wp::float32 var_830;
        wp::int32 var_831;
        wp::int32 var_832;
        wp::int32 var_833;
        wp::float32* var_834;
        wp::float32 var_835;
        wp::float32 var_836;
        wp::int32 var_837;
        const wp::int32 var_838 = 3;
        const wp::int32 var_839 = 1;
        wp::int32 var_840;
        wp::int32 var_841;
        wp::int32 var_842;
        wp::float32* var_843;
        wp::float32 var_844;
        wp::float32 var_845;
        const wp::int32 var_846 = 1;
        wp::int32 var_847;
        const wp::int32 var_848 = 6;
        wp::range_t var_849;
        wp::int32 var_850;
        wp::int32 var_851;
        wp::int32 var_852;
        wp::int32 var_853;
        wp::float32* var_854;
        wp::int32 var_855;
        wp::float32* var_856;
        wp::float32 var_857;
        wp::float32 var_858;
        wp::float32 var_859;
        wp::float32 var_860;
        wp::int32 var_861;
        wp::int32 var_862;
        wp::int32 var_863;
        wp::float32* var_864;
        wp::float32 var_865;
        wp::float32 var_866;
        wp::int32 var_867;
        const wp::int32 var_868 = 4;
        const wp::int32 var_869 = 1;
        wp::int32 var_870;
        wp::int32 var_871;
        wp::int32 var_872;
        wp::float32* var_873;
        wp::float32 var_874;
        wp::float32 var_875;
        const wp::int32 var_876 = 1;
        wp::int32 var_877;
        const wp::int32 var_878 = 6;
        wp::range_t var_879;
        wp::int32 var_880;
        wp::int32 var_881;
        wp::int32 var_882;
        wp::int32 var_883;
        wp::float32* var_884;
        wp::int32 var_885;
        wp::float32* var_886;
        wp::float32 var_887;
        wp::float32 var_888;
        wp::float32 var_889;
        wp::float32 var_890;
        wp::int32 var_891;
        wp::int32 var_892;
        wp::int32 var_893;
        wp::float32* var_894;
        wp::float32 var_895;
        wp::float32 var_896;
        wp::int32 var_897;
        const wp::int32 var_898 = 5;
        const wp::int32 var_899 = 1;
        wp::int32 var_900;
        wp::int32 var_901;
        wp::int32 var_902;
        wp::float32* var_903;
        wp::float32 var_904;
        wp::float32 var_905;
        const wp::int32 var_906 = 1;
        wp::int32 var_907;
        const wp::int32 var_908 = 6;
        wp::range_t var_909;
        wp::int32 var_910;
        wp::int32 var_911;
        wp::int32 var_912;
        wp::int32 var_913;
        wp::float32* var_914;
        wp::int32 var_915;
        wp::float32* var_916;
        wp::float32 var_917;
        wp::float32 var_918;
        wp::float32 var_919;
        wp::float32 var_920;
        wp::int32 var_921;
        wp::int32 var_922;
        wp::int32 var_923;
        wp::float32* var_924;
        wp::float32 var_925;
        wp::float32 var_926;
        wp::int32 var_927;
        wp::int32 var_928;
        //---------
        // forward
        // def kernel(                                                                            <L 3266>
        // worldid, blk = wp.tid()                                                                <L 3280>
        builtin_tid2d(var_0, var_1);
        // start = block_dof[blk]                                                                 <L 3281>
        var_2 = wp::address(var_block_dof, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // size = wp.static(block_size)                                                           <L 3282>
        // matrix_adr = M_rowadr[start]                                                           <L 3283>
        var_6 = wp::address(var_M_rowadr, var_3);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // factor_adr = qLD_block_adr[start]                                                      <L 3285>
        var_9 = wp::address(var_qLD_block_adr, var_3);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if factor_adr == Q_LD_BLOCK_COMPACT:                                                   <L 3286>
        var_13 = (var_10 == var_12);
        if (var_13) {
            // for i in range(wp.static(block_size)):                                             <L 3287>
            // inverse = 1.0 / M_in[worldid, matrix_adr + i]                                      <L 3288>
            var_16 = wp::add(var_7, var_14);
            var_17 = wp::address(var_M_in, var_0, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::div(var_15, var_19);
            // D_out[worldid, start + i] = inverse                                                <L 3289>
            var_20 = wp::add(var_3, var_14);
            wp::array_store(var_D_out, var_0, var_20, var_18);
            // x_out[worldid, start + i] = inverse * y[worldid, start + i]                        <L 3290>
            var_21 = wp::add(var_3, var_14);
            var_22 = wp::address(var_y, var_0, var_21);
            var_24 = wp::load(var_22);
            var_23 = wp::mul(var_18, var_24);
            var_25 = wp::add(var_3, var_14);
            wp::array_store(var_x_out, var_0, var_25, var_23);
            // inverse = 1.0 / M_in[worldid, matrix_adr + i]                                      <L 3288>
            var_28 = wp::add(var_7, var_26);
            var_29 = wp::address(var_M_in, var_0, var_28);
            var_31 = wp::load(var_29);
            var_30 = wp::div(var_27, var_31);
            // D_out[worldid, start + i] = inverse                                                <L 3289>
            var_32 = wp::add(var_3, var_26);
            wp::array_store(var_D_out, var_0, var_32, var_30);
            // x_out[worldid, start + i] = inverse * y[worldid, start + i]                        <L 3290>
            var_33 = wp::add(var_3, var_26);
            var_34 = wp::address(var_y, var_0, var_33);
            var_36 = wp::load(var_34);
            var_35 = wp::mul(var_30, var_36);
            var_37 = wp::add(var_3, var_26);
            wp::array_store(var_x_out, var_0, var_37, var_35);
            // inverse = 1.0 / M_in[worldid, matrix_adr + i]                                      <L 3288>
            var_40 = wp::add(var_7, var_38);
            var_41 = wp::address(var_M_in, var_0, var_40);
            var_43 = wp::load(var_41);
            var_42 = wp::div(var_39, var_43);
            // D_out[worldid, start + i] = inverse                                                <L 3289>
            var_44 = wp::add(var_3, var_38);
            wp::array_store(var_D_out, var_0, var_44, var_42);
            // x_out[worldid, start + i] = inverse * y[worldid, start + i]                        <L 3290>
            var_45 = wp::add(var_3, var_38);
            var_46 = wp::address(var_y, var_0, var_45);
            var_48 = wp::load(var_46);
            var_47 = wp::mul(var_42, var_48);
            var_49 = wp::add(var_3, var_38);
            wp::array_store(var_x_out, var_0, var_49, var_47);
            // inverse = 1.0 / M_in[worldid, matrix_adr + i]                                      <L 3288>
            var_52 = wp::add(var_7, var_50);
            var_53 = wp::address(var_M_in, var_0, var_52);
            var_55 = wp::load(var_53);
            var_54 = wp::div(var_51, var_55);
            // D_out[worldid, start + i] = inverse                                                <L 3289>
            var_56 = wp::add(var_3, var_50);
            wp::array_store(var_D_out, var_0, var_56, var_54);
            // x_out[worldid, start + i] = inverse * y[worldid, start + i]                        <L 3290>
            var_57 = wp::add(var_3, var_50);
            var_58 = wp::address(var_y, var_0, var_57);
            var_60 = wp::load(var_58);
            var_59 = wp::mul(var_54, var_60);
            var_61 = wp::add(var_3, var_50);
            wp::array_store(var_x_out, var_0, var_61, var_59);
            // inverse = 1.0 / M_in[worldid, matrix_adr + i]                                      <L 3288>
            var_64 = wp::add(var_7, var_62);
            var_65 = wp::address(var_M_in, var_0, var_64);
            var_67 = wp::load(var_65);
            var_66 = wp::div(var_63, var_67);
            // D_out[worldid, start + i] = inverse                                                <L 3289>
            var_68 = wp::add(var_3, var_62);
            wp::array_store(var_D_out, var_0, var_68, var_66);
            // x_out[worldid, start + i] = inverse * y[worldid, start + i]                        <L 3290>
            var_69 = wp::add(var_3, var_62);
            var_70 = wp::address(var_y, var_0, var_69);
            var_72 = wp::load(var_70);
            var_71 = wp::mul(var_66, var_72);
            var_73 = wp::add(var_3, var_62);
            wp::array_store(var_x_out, var_0, var_73, var_71);
            // inverse = 1.0 / M_in[worldid, matrix_adr + i]                                      <L 3288>
            var_76 = wp::add(var_7, var_74);
            var_77 = wp::address(var_M_in, var_0, var_76);
            var_79 = wp::load(var_77);
            var_78 = wp::div(var_75, var_79);
            // D_out[worldid, start + i] = inverse                                                <L 3289>
            var_80 = wp::add(var_3, var_74);
            wp::array_store(var_D_out, var_0, var_80, var_78);
            // x_out[worldid, start + i] = inverse * y[worldid, start + i]                        <L 3290>
            var_81 = wp::add(var_3, var_74);
            var_82 = wp::address(var_y, var_0, var_81);
            var_84 = wp::load(var_82);
            var_83 = wp::mul(var_78, var_84);
            var_85 = wp::add(var_3, var_74);
            wp::array_store(var_x_out, var_0, var_85, var_83);
        }
        if (!var_13) {
            // for i in range(wp.static(block_size)):                                             <L 3292>
            // diagonal_value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                  <L 3293>
            var_88 = wp::add(var_86, var_87);
            var_89 = wp::mul(var_86, var_88);
            var_91 = wp::floordiv(var_89, var_90);
            var_92 = wp::add(var_7, var_91);
            var_93 = wp::add(var_92, var_86);
            var_94 = wp::address(var_M_in, var_0, var_93);
            var_96 = wp::load(var_94);
            var_95 = wp::copy(var_96);
            // rhs_value = y[worldid, start + i]                                                  <L 3294>
            var_97 = wp::add(var_3, var_86);
            var_98 = wp::address(var_y, var_0, var_97);
            var_100 = wp::load(var_98);
            var_99 = wp::copy(var_100);
            // for k in range(i):                                                                 <L 3295>
            // diagonal_factor = wp.sqrt(diagonal_value)                                          <L 3300>
            var_101 = wp::sqrt(var_95);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_factor                        <L 3301>
            var_102 = wp::mul(var_86, var_5);
            var_103 = wp::add(var_10, var_102);
            var_104 = wp::add(var_103, var_86);
            wp::array_store(var_L_out, var_0, var_104, var_101);
            // diagonal_inv = 1.0 / diagonal_factor                                               <L 3302>
            var_106 = wp::div(var_105, var_101);
            // x_out[worldid, start + i] = rhs_value * diagonal_inv                               <L 3303>
            var_107 = wp::mul(var_99, var_106);
            var_108 = wp::add(var_3, var_86);
            wp::array_store(var_x_out, var_0, var_108, var_107);
            // for j in range(i + 1, size):                                                       <L 3305>
            var_110 = wp::add(var_86, var_109);
            var_112 = wp::range(var_110, var_111);
            start_for_0:;
                if (iter_cmp(var_112) == 0) goto end_for_0;
                var_113 = wp::iter_next(var_112);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 3306>
                var_115 = wp::add(var_113, var_114);
                var_116 = wp::mul(var_113, var_115);
                var_118 = wp::floordiv(var_116, var_117);
                var_119 = wp::add(var_7, var_118);
                var_120 = wp::add(var_119, var_86);
                var_121 = wp::address(var_M_in, var_0, var_120);
                var_123 = wp::load(var_121);
                var_122 = wp::copy(var_123);
                // for k in range(i):                                                             <L 3307>
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 3309>
                var_124 = wp::mul(var_122, var_106);
                var_125 = wp::mul(var_86, var_5);
                var_126 = wp::add(var_10, var_125);
                var_127 = wp::add(var_126, var_113);
                wp::array_store(var_L_out, var_0, var_127, var_124);
                goto start_for_0;
            end_for_0:;
            // diagonal_value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                  <L 3293>
            var_130 = wp::add(var_128, var_129);
            var_131 = wp::mul(var_128, var_130);
            var_133 = wp::floordiv(var_131, var_132);
            var_134 = wp::add(var_7, var_133);
            var_135 = wp::add(var_134, var_128);
            var_136 = wp::address(var_M_in, var_0, var_135);
            var_138 = wp::load(var_136);
            var_137 = wp::copy(var_138);
            // rhs_value = y[worldid, start + i]                                                  <L 3294>
            var_139 = wp::add(var_3, var_128);
            var_140 = wp::address(var_y, var_0, var_139);
            var_142 = wp::load(var_140);
            var_141 = wp::copy(var_142);
            // for k in range(i):                                                                 <L 3295>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_144 = wp::mul(var_143, var_5);
            var_145 = wp::add(var_10, var_144);
            var_146 = wp::add(var_145, var_128);
            var_147 = wp::address(var_L_out, var_0, var_146);
            var_149 = wp::load(var_147);
            var_148 = wp::copy(var_149);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_150 = wp::mul(var_148, var_148);
            var_151 = wp::sub(var_137, var_150);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_152 = wp::add(var_3, var_143);
            var_153 = wp::address(var_x_out, var_0, var_152);
            var_155 = wp::load(var_153);
            var_154 = wp::mul(var_148, var_155);
            var_156 = wp::sub(var_141, var_154);
            // diagonal_factor = wp.sqrt(diagonal_value)                                          <L 3300>
            var_157 = wp::sqrt(var_151);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_factor                        <L 3301>
            var_158 = wp::mul(var_128, var_5);
            var_159 = wp::add(var_10, var_158);
            var_160 = wp::add(var_159, var_128);
            wp::array_store(var_L_out, var_0, var_160, var_157);
            // diagonal_inv = 1.0 / diagonal_factor                                               <L 3302>
            var_162 = wp::div(var_161, var_157);
            // x_out[worldid, start + i] = rhs_value * diagonal_inv                               <L 3303>
            var_163 = wp::mul(var_156, var_162);
            var_164 = wp::add(var_3, var_128);
            wp::array_store(var_x_out, var_0, var_164, var_163);
            // for j in range(i + 1, size):                                                       <L 3305>
            var_166 = wp::add(var_128, var_165);
            var_168 = wp::range(var_166, var_167);
            var_169 = wp::copy(var_143);
            start_for_2:;
                if (iter_cmp(var_168) == 0) goto end_for_2;
                var_170 = wp::iter_next(var_168);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 3306>
                var_172 = wp::add(var_170, var_171);
                var_173 = wp::mul(var_170, var_172);
                var_175 = wp::floordiv(var_173, var_174);
                var_176 = wp::add(var_7, var_175);
                var_177 = wp::add(var_176, var_128);
                var_178 = wp::address(var_M_in, var_0, var_177);
                var_180 = wp::load(var_178);
                var_179 = wp::copy(var_180);
                // for k in range(i):                                                             <L 3307>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 3308>
                var_182 = wp::mul(var_181, var_5);
                var_183 = wp::add(var_10, var_182);
                var_184 = wp::add(var_183, var_128);
                var_185 = wp::address(var_L_out, var_0, var_184);
                var_186 = wp::mul(var_181, var_5);
                var_187 = wp::add(var_10, var_186);
                var_188 = wp::add(var_187, var_170);
                var_189 = wp::address(var_L_out, var_0, var_188);
                var_191 = wp::load(var_185);
                var_192 = wp::load(var_189);
                var_190 = wp::mul(var_191, var_192);
                var_193 = wp::sub(var_179, var_190);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 3309>
                var_194 = wp::mul(var_193, var_162);
                var_195 = wp::mul(var_128, var_5);
                var_196 = wp::add(var_10, var_195);
                var_197 = wp::add(var_196, var_170);
                wp::array_store(var_L_out, var_0, var_197, var_194);
                wp::assign(var_122, var_193);
                wp::assign(var_169, var_181);
                goto start_for_2;
            end_for_2:;
            // diagonal_value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                  <L 3293>
            var_200 = wp::add(var_198, var_199);
            var_201 = wp::mul(var_198, var_200);
            var_203 = wp::floordiv(var_201, var_202);
            var_204 = wp::add(var_7, var_203);
            var_205 = wp::add(var_204, var_198);
            var_206 = wp::address(var_M_in, var_0, var_205);
            var_208 = wp::load(var_206);
            var_207 = wp::copy(var_208);
            // rhs_value = y[worldid, start + i]                                                  <L 3294>
            var_209 = wp::add(var_3, var_198);
            var_210 = wp::address(var_y, var_0, var_209);
            var_212 = wp::load(var_210);
            var_211 = wp::copy(var_212);
            // for k in range(i):                                                                 <L 3295>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_214 = wp::mul(var_213, var_5);
            var_215 = wp::add(var_10, var_214);
            var_216 = wp::add(var_215, var_198);
            var_217 = wp::address(var_L_out, var_0, var_216);
            var_219 = wp::load(var_217);
            var_218 = wp::copy(var_219);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_220 = wp::mul(var_218, var_218);
            var_221 = wp::sub(var_207, var_220);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_222 = wp::add(var_3, var_213);
            var_223 = wp::address(var_x_out, var_0, var_222);
            var_225 = wp::load(var_223);
            var_224 = wp::mul(var_218, var_225);
            var_226 = wp::sub(var_211, var_224);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_228 = wp::mul(var_227, var_5);
            var_229 = wp::add(var_10, var_228);
            var_230 = wp::add(var_229, var_198);
            var_231 = wp::address(var_L_out, var_0, var_230);
            var_233 = wp::load(var_231);
            var_232 = wp::copy(var_233);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_234 = wp::mul(var_232, var_232);
            var_235 = wp::sub(var_221, var_234);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_236 = wp::add(var_3, var_227);
            var_237 = wp::address(var_x_out, var_0, var_236);
            var_239 = wp::load(var_237);
            var_238 = wp::mul(var_232, var_239);
            var_240 = wp::sub(var_226, var_238);
            // diagonal_factor = wp.sqrt(diagonal_value)                                          <L 3300>
            var_241 = wp::sqrt(var_235);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_factor                        <L 3301>
            var_242 = wp::mul(var_198, var_5);
            var_243 = wp::add(var_10, var_242);
            var_244 = wp::add(var_243, var_198);
            wp::array_store(var_L_out, var_0, var_244, var_241);
            // diagonal_inv = 1.0 / diagonal_factor                                               <L 3302>
            var_246 = wp::div(var_245, var_241);
            // x_out[worldid, start + i] = rhs_value * diagonal_inv                               <L 3303>
            var_247 = wp::mul(var_240, var_246);
            var_248 = wp::add(var_3, var_198);
            wp::array_store(var_x_out, var_0, var_248, var_247);
            // for j in range(i + 1, size):                                                       <L 3305>
            var_250 = wp::add(var_198, var_249);
            var_252 = wp::range(var_250, var_251);
            var_253 = wp::copy(var_227);
            start_for_4:;
                if (iter_cmp(var_252) == 0) goto end_for_4;
                var_254 = wp::iter_next(var_252);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 3306>
                var_256 = wp::add(var_254, var_255);
                var_257 = wp::mul(var_254, var_256);
                var_259 = wp::floordiv(var_257, var_258);
                var_260 = wp::add(var_7, var_259);
                var_261 = wp::add(var_260, var_198);
                var_262 = wp::address(var_M_in, var_0, var_261);
                var_264 = wp::load(var_262);
                var_263 = wp::copy(var_264);
                // for k in range(i):                                                             <L 3307>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 3308>
                var_266 = wp::mul(var_265, var_5);
                var_267 = wp::add(var_10, var_266);
                var_268 = wp::add(var_267, var_198);
                var_269 = wp::address(var_L_out, var_0, var_268);
                var_270 = wp::mul(var_265, var_5);
                var_271 = wp::add(var_10, var_270);
                var_272 = wp::add(var_271, var_254);
                var_273 = wp::address(var_L_out, var_0, var_272);
                var_275 = wp::load(var_269);
                var_276 = wp::load(var_273);
                var_274 = wp::mul(var_275, var_276);
                var_277 = wp::sub(var_263, var_274);
                var_279 = wp::mul(var_278, var_5);
                var_280 = wp::add(var_10, var_279);
                var_281 = wp::add(var_280, var_198);
                var_282 = wp::address(var_L_out, var_0, var_281);
                var_283 = wp::mul(var_278, var_5);
                var_284 = wp::add(var_10, var_283);
                var_285 = wp::add(var_284, var_254);
                var_286 = wp::address(var_L_out, var_0, var_285);
                var_288 = wp::load(var_282);
                var_289 = wp::load(var_286);
                var_287 = wp::mul(var_288, var_289);
                var_290 = wp::sub(var_277, var_287);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 3309>
                var_291 = wp::mul(var_290, var_246);
                var_292 = wp::mul(var_198, var_5);
                var_293 = wp::add(var_10, var_292);
                var_294 = wp::add(var_293, var_254);
                wp::array_store(var_L_out, var_0, var_294, var_291);
                wp::assign(var_122, var_290);
                wp::assign(var_253, var_278);
                goto start_for_4;
            end_for_4:;
            // diagonal_value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                  <L 3293>
            var_297 = wp::add(var_295, var_296);
            var_298 = wp::mul(var_295, var_297);
            var_300 = wp::floordiv(var_298, var_299);
            var_301 = wp::add(var_7, var_300);
            var_302 = wp::add(var_301, var_295);
            var_303 = wp::address(var_M_in, var_0, var_302);
            var_305 = wp::load(var_303);
            var_304 = wp::copy(var_305);
            // rhs_value = y[worldid, start + i]                                                  <L 3294>
            var_306 = wp::add(var_3, var_295);
            var_307 = wp::address(var_y, var_0, var_306);
            var_309 = wp::load(var_307);
            var_308 = wp::copy(var_309);
            // for k in range(i):                                                                 <L 3295>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_311 = wp::mul(var_310, var_5);
            var_312 = wp::add(var_10, var_311);
            var_313 = wp::add(var_312, var_295);
            var_314 = wp::address(var_L_out, var_0, var_313);
            var_316 = wp::load(var_314);
            var_315 = wp::copy(var_316);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_317 = wp::mul(var_315, var_315);
            var_318 = wp::sub(var_304, var_317);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_319 = wp::add(var_3, var_310);
            var_320 = wp::address(var_x_out, var_0, var_319);
            var_322 = wp::load(var_320);
            var_321 = wp::mul(var_315, var_322);
            var_323 = wp::sub(var_308, var_321);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_325 = wp::mul(var_324, var_5);
            var_326 = wp::add(var_10, var_325);
            var_327 = wp::add(var_326, var_295);
            var_328 = wp::address(var_L_out, var_0, var_327);
            var_330 = wp::load(var_328);
            var_329 = wp::copy(var_330);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_331 = wp::mul(var_329, var_329);
            var_332 = wp::sub(var_318, var_331);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_333 = wp::add(var_3, var_324);
            var_334 = wp::address(var_x_out, var_0, var_333);
            var_336 = wp::load(var_334);
            var_335 = wp::mul(var_329, var_336);
            var_337 = wp::sub(var_323, var_335);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_339 = wp::mul(var_338, var_5);
            var_340 = wp::add(var_10, var_339);
            var_341 = wp::add(var_340, var_295);
            var_342 = wp::address(var_L_out, var_0, var_341);
            var_344 = wp::load(var_342);
            var_343 = wp::copy(var_344);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_345 = wp::mul(var_343, var_343);
            var_346 = wp::sub(var_332, var_345);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_347 = wp::add(var_3, var_338);
            var_348 = wp::address(var_x_out, var_0, var_347);
            var_350 = wp::load(var_348);
            var_349 = wp::mul(var_343, var_350);
            var_351 = wp::sub(var_337, var_349);
            // diagonal_factor = wp.sqrt(diagonal_value)                                          <L 3300>
            var_352 = wp::sqrt(var_346);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_factor                        <L 3301>
            var_353 = wp::mul(var_295, var_5);
            var_354 = wp::add(var_10, var_353);
            var_355 = wp::add(var_354, var_295);
            wp::array_store(var_L_out, var_0, var_355, var_352);
            // diagonal_inv = 1.0 / diagonal_factor                                               <L 3302>
            var_357 = wp::div(var_356, var_352);
            // x_out[worldid, start + i] = rhs_value * diagonal_inv                               <L 3303>
            var_358 = wp::mul(var_351, var_357);
            var_359 = wp::add(var_3, var_295);
            wp::array_store(var_x_out, var_0, var_359, var_358);
            // for j in range(i + 1, size):                                                       <L 3305>
            var_361 = wp::add(var_295, var_360);
            var_363 = wp::range(var_361, var_362);
            var_364 = wp::copy(var_338);
            start_for_6:;
                if (iter_cmp(var_363) == 0) goto end_for_6;
                var_365 = wp::iter_next(var_363);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 3306>
                var_367 = wp::add(var_365, var_366);
                var_368 = wp::mul(var_365, var_367);
                var_370 = wp::floordiv(var_368, var_369);
                var_371 = wp::add(var_7, var_370);
                var_372 = wp::add(var_371, var_295);
                var_373 = wp::address(var_M_in, var_0, var_372);
                var_375 = wp::load(var_373);
                var_374 = wp::copy(var_375);
                // for k in range(i):                                                             <L 3307>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 3308>
                var_377 = wp::mul(var_376, var_5);
                var_378 = wp::add(var_10, var_377);
                var_379 = wp::add(var_378, var_295);
                var_380 = wp::address(var_L_out, var_0, var_379);
                var_381 = wp::mul(var_376, var_5);
                var_382 = wp::add(var_10, var_381);
                var_383 = wp::add(var_382, var_365);
                var_384 = wp::address(var_L_out, var_0, var_383);
                var_386 = wp::load(var_380);
                var_387 = wp::load(var_384);
                var_385 = wp::mul(var_386, var_387);
                var_388 = wp::sub(var_374, var_385);
                var_390 = wp::mul(var_389, var_5);
                var_391 = wp::add(var_10, var_390);
                var_392 = wp::add(var_391, var_295);
                var_393 = wp::address(var_L_out, var_0, var_392);
                var_394 = wp::mul(var_389, var_5);
                var_395 = wp::add(var_10, var_394);
                var_396 = wp::add(var_395, var_365);
                var_397 = wp::address(var_L_out, var_0, var_396);
                var_399 = wp::load(var_393);
                var_400 = wp::load(var_397);
                var_398 = wp::mul(var_399, var_400);
                var_401 = wp::sub(var_388, var_398);
                var_403 = wp::mul(var_402, var_5);
                var_404 = wp::add(var_10, var_403);
                var_405 = wp::add(var_404, var_295);
                var_406 = wp::address(var_L_out, var_0, var_405);
                var_407 = wp::mul(var_402, var_5);
                var_408 = wp::add(var_10, var_407);
                var_409 = wp::add(var_408, var_365);
                var_410 = wp::address(var_L_out, var_0, var_409);
                var_412 = wp::load(var_406);
                var_413 = wp::load(var_410);
                var_411 = wp::mul(var_412, var_413);
                var_414 = wp::sub(var_401, var_411);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 3309>
                var_415 = wp::mul(var_414, var_357);
                var_416 = wp::mul(var_295, var_5);
                var_417 = wp::add(var_10, var_416);
                var_418 = wp::add(var_417, var_365);
                wp::array_store(var_L_out, var_0, var_418, var_415);
                wp::assign(var_122, var_414);
                wp::assign(var_364, var_402);
                goto start_for_6;
            end_for_6:;
            // diagonal_value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                  <L 3293>
            var_421 = wp::add(var_419, var_420);
            var_422 = wp::mul(var_419, var_421);
            var_424 = wp::floordiv(var_422, var_423);
            var_425 = wp::add(var_7, var_424);
            var_426 = wp::add(var_425, var_419);
            var_427 = wp::address(var_M_in, var_0, var_426);
            var_429 = wp::load(var_427);
            var_428 = wp::copy(var_429);
            // rhs_value = y[worldid, start + i]                                                  <L 3294>
            var_430 = wp::add(var_3, var_419);
            var_431 = wp::address(var_y, var_0, var_430);
            var_433 = wp::load(var_431);
            var_432 = wp::copy(var_433);
            // for k in range(i):                                                                 <L 3295>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_435 = wp::mul(var_434, var_5);
            var_436 = wp::add(var_10, var_435);
            var_437 = wp::add(var_436, var_419);
            var_438 = wp::address(var_L_out, var_0, var_437);
            var_440 = wp::load(var_438);
            var_439 = wp::copy(var_440);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_441 = wp::mul(var_439, var_439);
            var_442 = wp::sub(var_428, var_441);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_443 = wp::add(var_3, var_434);
            var_444 = wp::address(var_x_out, var_0, var_443);
            var_446 = wp::load(var_444);
            var_445 = wp::mul(var_439, var_446);
            var_447 = wp::sub(var_432, var_445);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_449 = wp::mul(var_448, var_5);
            var_450 = wp::add(var_10, var_449);
            var_451 = wp::add(var_450, var_419);
            var_452 = wp::address(var_L_out, var_0, var_451);
            var_454 = wp::load(var_452);
            var_453 = wp::copy(var_454);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_455 = wp::mul(var_453, var_453);
            var_456 = wp::sub(var_442, var_455);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_457 = wp::add(var_3, var_448);
            var_458 = wp::address(var_x_out, var_0, var_457);
            var_460 = wp::load(var_458);
            var_459 = wp::mul(var_453, var_460);
            var_461 = wp::sub(var_447, var_459);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_463 = wp::mul(var_462, var_5);
            var_464 = wp::add(var_10, var_463);
            var_465 = wp::add(var_464, var_419);
            var_466 = wp::address(var_L_out, var_0, var_465);
            var_468 = wp::load(var_466);
            var_467 = wp::copy(var_468);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_469 = wp::mul(var_467, var_467);
            var_470 = wp::sub(var_456, var_469);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_471 = wp::add(var_3, var_462);
            var_472 = wp::address(var_x_out, var_0, var_471);
            var_474 = wp::load(var_472);
            var_473 = wp::mul(var_467, var_474);
            var_475 = wp::sub(var_461, var_473);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_477 = wp::mul(var_476, var_5);
            var_478 = wp::add(var_10, var_477);
            var_479 = wp::add(var_478, var_419);
            var_480 = wp::address(var_L_out, var_0, var_479);
            var_482 = wp::load(var_480);
            var_481 = wp::copy(var_482);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_483 = wp::mul(var_481, var_481);
            var_484 = wp::sub(var_470, var_483);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_485 = wp::add(var_3, var_476);
            var_486 = wp::address(var_x_out, var_0, var_485);
            var_488 = wp::load(var_486);
            var_487 = wp::mul(var_481, var_488);
            var_489 = wp::sub(var_475, var_487);
            // diagonal_factor = wp.sqrt(diagonal_value)                                          <L 3300>
            var_490 = wp::sqrt(var_484);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_factor                        <L 3301>
            var_491 = wp::mul(var_419, var_5);
            var_492 = wp::add(var_10, var_491);
            var_493 = wp::add(var_492, var_419);
            wp::array_store(var_L_out, var_0, var_493, var_490);
            // diagonal_inv = 1.0 / diagonal_factor                                               <L 3302>
            var_495 = wp::div(var_494, var_490);
            // x_out[worldid, start + i] = rhs_value * diagonal_inv                               <L 3303>
            var_496 = wp::mul(var_489, var_495);
            var_497 = wp::add(var_3, var_419);
            wp::array_store(var_x_out, var_0, var_497, var_496);
            // for j in range(i + 1, size):                                                       <L 3305>
            var_499 = wp::add(var_419, var_498);
            var_501 = wp::range(var_499, var_500);
            var_502 = wp::copy(var_476);
            start_for_8:;
                if (iter_cmp(var_501) == 0) goto end_for_8;
                var_503 = wp::iter_next(var_501);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 3306>
                var_505 = wp::add(var_503, var_504);
                var_506 = wp::mul(var_503, var_505);
                var_508 = wp::floordiv(var_506, var_507);
                var_509 = wp::add(var_7, var_508);
                var_510 = wp::add(var_509, var_419);
                var_511 = wp::address(var_M_in, var_0, var_510);
                var_513 = wp::load(var_511);
                var_512 = wp::copy(var_513);
                // for k in range(i):                                                             <L 3307>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 3308>
                var_515 = wp::mul(var_514, var_5);
                var_516 = wp::add(var_10, var_515);
                var_517 = wp::add(var_516, var_419);
                var_518 = wp::address(var_L_out, var_0, var_517);
                var_519 = wp::mul(var_514, var_5);
                var_520 = wp::add(var_10, var_519);
                var_521 = wp::add(var_520, var_503);
                var_522 = wp::address(var_L_out, var_0, var_521);
                var_524 = wp::load(var_518);
                var_525 = wp::load(var_522);
                var_523 = wp::mul(var_524, var_525);
                var_526 = wp::sub(var_512, var_523);
                var_528 = wp::mul(var_527, var_5);
                var_529 = wp::add(var_10, var_528);
                var_530 = wp::add(var_529, var_419);
                var_531 = wp::address(var_L_out, var_0, var_530);
                var_532 = wp::mul(var_527, var_5);
                var_533 = wp::add(var_10, var_532);
                var_534 = wp::add(var_533, var_503);
                var_535 = wp::address(var_L_out, var_0, var_534);
                var_537 = wp::load(var_531);
                var_538 = wp::load(var_535);
                var_536 = wp::mul(var_537, var_538);
                var_539 = wp::sub(var_526, var_536);
                var_541 = wp::mul(var_540, var_5);
                var_542 = wp::add(var_10, var_541);
                var_543 = wp::add(var_542, var_419);
                var_544 = wp::address(var_L_out, var_0, var_543);
                var_545 = wp::mul(var_540, var_5);
                var_546 = wp::add(var_10, var_545);
                var_547 = wp::add(var_546, var_503);
                var_548 = wp::address(var_L_out, var_0, var_547);
                var_550 = wp::load(var_544);
                var_551 = wp::load(var_548);
                var_549 = wp::mul(var_550, var_551);
                var_552 = wp::sub(var_539, var_549);
                var_554 = wp::mul(var_553, var_5);
                var_555 = wp::add(var_10, var_554);
                var_556 = wp::add(var_555, var_419);
                var_557 = wp::address(var_L_out, var_0, var_556);
                var_558 = wp::mul(var_553, var_5);
                var_559 = wp::add(var_10, var_558);
                var_560 = wp::add(var_559, var_503);
                var_561 = wp::address(var_L_out, var_0, var_560);
                var_563 = wp::load(var_557);
                var_564 = wp::load(var_561);
                var_562 = wp::mul(var_563, var_564);
                var_565 = wp::sub(var_552, var_562);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 3309>
                var_566 = wp::mul(var_565, var_495);
                var_567 = wp::mul(var_419, var_5);
                var_568 = wp::add(var_10, var_567);
                var_569 = wp::add(var_568, var_503);
                wp::array_store(var_L_out, var_0, var_569, var_566);
                wp::assign(var_122, var_565);
                wp::assign(var_502, var_553);
                goto start_for_8;
            end_for_8:;
            // diagonal_value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                  <L 3293>
            var_572 = wp::add(var_570, var_571);
            var_573 = wp::mul(var_570, var_572);
            var_575 = wp::floordiv(var_573, var_574);
            var_576 = wp::add(var_7, var_575);
            var_577 = wp::add(var_576, var_570);
            var_578 = wp::address(var_M_in, var_0, var_577);
            var_580 = wp::load(var_578);
            var_579 = wp::copy(var_580);
            // rhs_value = y[worldid, start + i]                                                  <L 3294>
            var_581 = wp::add(var_3, var_570);
            var_582 = wp::address(var_y, var_0, var_581);
            var_584 = wp::load(var_582);
            var_583 = wp::copy(var_584);
            // for k in range(i):                                                                 <L 3295>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_586 = wp::mul(var_585, var_5);
            var_587 = wp::add(var_10, var_586);
            var_588 = wp::add(var_587, var_570);
            var_589 = wp::address(var_L_out, var_0, var_588);
            var_591 = wp::load(var_589);
            var_590 = wp::copy(var_591);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_592 = wp::mul(var_590, var_590);
            var_593 = wp::sub(var_579, var_592);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_594 = wp::add(var_3, var_585);
            var_595 = wp::address(var_x_out, var_0, var_594);
            var_597 = wp::load(var_595);
            var_596 = wp::mul(var_590, var_597);
            var_598 = wp::sub(var_583, var_596);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_600 = wp::mul(var_599, var_5);
            var_601 = wp::add(var_10, var_600);
            var_602 = wp::add(var_601, var_570);
            var_603 = wp::address(var_L_out, var_0, var_602);
            var_605 = wp::load(var_603);
            var_604 = wp::copy(var_605);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_606 = wp::mul(var_604, var_604);
            var_607 = wp::sub(var_593, var_606);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_608 = wp::add(var_3, var_599);
            var_609 = wp::address(var_x_out, var_0, var_608);
            var_611 = wp::load(var_609);
            var_610 = wp::mul(var_604, var_611);
            var_612 = wp::sub(var_598, var_610);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_614 = wp::mul(var_613, var_5);
            var_615 = wp::add(var_10, var_614);
            var_616 = wp::add(var_615, var_570);
            var_617 = wp::address(var_L_out, var_0, var_616);
            var_619 = wp::load(var_617);
            var_618 = wp::copy(var_619);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_620 = wp::mul(var_618, var_618);
            var_621 = wp::sub(var_607, var_620);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_622 = wp::add(var_3, var_613);
            var_623 = wp::address(var_x_out, var_0, var_622);
            var_625 = wp::load(var_623);
            var_624 = wp::mul(var_618, var_625);
            var_626 = wp::sub(var_612, var_624);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_628 = wp::mul(var_627, var_5);
            var_629 = wp::add(var_10, var_628);
            var_630 = wp::add(var_629, var_570);
            var_631 = wp::address(var_L_out, var_0, var_630);
            var_633 = wp::load(var_631);
            var_632 = wp::copy(var_633);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_634 = wp::mul(var_632, var_632);
            var_635 = wp::sub(var_621, var_634);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_636 = wp::add(var_3, var_627);
            var_637 = wp::address(var_x_out, var_0, var_636);
            var_639 = wp::load(var_637);
            var_638 = wp::mul(var_632, var_639);
            var_640 = wp::sub(var_626, var_638);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 3296>
            var_642 = wp::mul(var_641, var_5);
            var_643 = wp::add(var_10, var_642);
            var_644 = wp::add(var_643, var_570);
            var_645 = wp::address(var_L_out, var_0, var_644);
            var_647 = wp::load(var_645);
            var_646 = wp::copy(var_647);
            // diagonal_value -= factor * factor                                                  <L 3297>
            var_648 = wp::mul(var_646, var_646);
            var_649 = wp::sub(var_635, var_648);
            // rhs_value -= factor * x_out[worldid, start + k]                                    <L 3298>
            var_650 = wp::add(var_3, var_641);
            var_651 = wp::address(var_x_out, var_0, var_650);
            var_653 = wp::load(var_651);
            var_652 = wp::mul(var_646, var_653);
            var_654 = wp::sub(var_640, var_652);
            // diagonal_factor = wp.sqrt(diagonal_value)                                          <L 3300>
            var_655 = wp::sqrt(var_649);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_factor                        <L 3301>
            var_656 = wp::mul(var_570, var_5);
            var_657 = wp::add(var_10, var_656);
            var_658 = wp::add(var_657, var_570);
            wp::array_store(var_L_out, var_0, var_658, var_655);
            // diagonal_inv = 1.0 / diagonal_factor                                               <L 3302>
            var_660 = wp::div(var_659, var_655);
            // x_out[worldid, start + i] = rhs_value * diagonal_inv                               <L 3303>
            var_661 = wp::mul(var_654, var_660);
            var_662 = wp::add(var_3, var_570);
            wp::array_store(var_x_out, var_0, var_662, var_661);
            // for j in range(i + 1, size):                                                       <L 3305>
            var_664 = wp::add(var_570, var_663);
            var_666 = wp::range(var_664, var_665);
            var_667 = wp::copy(var_641);
            start_for_10:;
                if (iter_cmp(var_666) == 0) goto end_for_10;
                var_668 = wp::iter_next(var_666);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 3306>
                var_670 = wp::add(var_668, var_669);
                var_671 = wp::mul(var_668, var_670);
                var_673 = wp::floordiv(var_671, var_672);
                var_674 = wp::add(var_7, var_673);
                var_675 = wp::add(var_674, var_570);
                var_676 = wp::address(var_M_in, var_0, var_675);
                var_678 = wp::load(var_676);
                var_677 = wp::copy(var_678);
                // for k in range(i):                                                             <L 3307>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 3308>
                var_680 = wp::mul(var_679, var_5);
                var_681 = wp::add(var_10, var_680);
                var_682 = wp::add(var_681, var_570);
                var_683 = wp::address(var_L_out, var_0, var_682);
                var_684 = wp::mul(var_679, var_5);
                var_685 = wp::add(var_10, var_684);
                var_686 = wp::add(var_685, var_668);
                var_687 = wp::address(var_L_out, var_0, var_686);
                var_689 = wp::load(var_683);
                var_690 = wp::load(var_687);
                var_688 = wp::mul(var_689, var_690);
                var_691 = wp::sub(var_677, var_688);
                var_693 = wp::mul(var_692, var_5);
                var_694 = wp::add(var_10, var_693);
                var_695 = wp::add(var_694, var_570);
                var_696 = wp::address(var_L_out, var_0, var_695);
                var_697 = wp::mul(var_692, var_5);
                var_698 = wp::add(var_10, var_697);
                var_699 = wp::add(var_698, var_668);
                var_700 = wp::address(var_L_out, var_0, var_699);
                var_702 = wp::load(var_696);
                var_703 = wp::load(var_700);
                var_701 = wp::mul(var_702, var_703);
                var_704 = wp::sub(var_691, var_701);
                var_706 = wp::mul(var_705, var_5);
                var_707 = wp::add(var_10, var_706);
                var_708 = wp::add(var_707, var_570);
                var_709 = wp::address(var_L_out, var_0, var_708);
                var_710 = wp::mul(var_705, var_5);
                var_711 = wp::add(var_10, var_710);
                var_712 = wp::add(var_711, var_668);
                var_713 = wp::address(var_L_out, var_0, var_712);
                var_715 = wp::load(var_709);
                var_716 = wp::load(var_713);
                var_714 = wp::mul(var_715, var_716);
                var_717 = wp::sub(var_704, var_714);
                var_719 = wp::mul(var_718, var_5);
                var_720 = wp::add(var_10, var_719);
                var_721 = wp::add(var_720, var_570);
                var_722 = wp::address(var_L_out, var_0, var_721);
                var_723 = wp::mul(var_718, var_5);
                var_724 = wp::add(var_10, var_723);
                var_725 = wp::add(var_724, var_668);
                var_726 = wp::address(var_L_out, var_0, var_725);
                var_728 = wp::load(var_722);
                var_729 = wp::load(var_726);
                var_727 = wp::mul(var_728, var_729);
                var_730 = wp::sub(var_717, var_727);
                var_732 = wp::mul(var_731, var_5);
                var_733 = wp::add(var_10, var_732);
                var_734 = wp::add(var_733, var_570);
                var_735 = wp::address(var_L_out, var_0, var_734);
                var_736 = wp::mul(var_731, var_5);
                var_737 = wp::add(var_10, var_736);
                var_738 = wp::add(var_737, var_668);
                var_739 = wp::address(var_L_out, var_0, var_738);
                var_741 = wp::load(var_735);
                var_742 = wp::load(var_739);
                var_740 = wp::mul(var_741, var_742);
                var_743 = wp::sub(var_730, var_740);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 3309>
                var_744 = wp::mul(var_743, var_660);
                var_745 = wp::mul(var_570, var_5);
                var_746 = wp::add(var_10, var_745);
                var_747 = wp::add(var_746, var_668);
                wp::array_store(var_L_out, var_0, var_747, var_744);
                wp::assign(var_122, var_743);
                wp::assign(var_667, var_731);
                goto start_for_10;
            end_for_10:;
            // for reverse_i in range(wp.static(block_size)):                                     <L 3311>
            // i = size - 1 - reverse_i                                                           <L 3312>
            var_750 = wp::sub(var_5, var_749);
            var_751 = wp::sub(var_750, var_748);
            // value = x_out[worldid, start + i]                                                  <L 3313>
            var_752 = wp::add(var_3, var_751);
            var_753 = wp::address(var_x_out, var_0, var_752);
            var_755 = wp::load(var_753);
            var_754 = wp::copy(var_755);
            // for k in range(i + 1, size):                                                       <L 3314>
            var_757 = wp::add(var_751, var_756);
            var_759 = wp::range(var_757, var_758);
            start_for_12:;
                if (iter_cmp(var_759) == 0) goto end_for_12;
                var_760 = wp::iter_next(var_759);
                // value -= L_out[worldid, factor_adr + i * size + k] * x_out[worldid, start + k]       <L 3315>
                var_761 = wp::mul(var_751, var_5);
                var_762 = wp::add(var_10, var_761);
                var_763 = wp::add(var_762, var_760);
                var_764 = wp::address(var_L_out, var_0, var_763);
                var_765 = wp::add(var_3, var_760);
                var_766 = wp::address(var_x_out, var_0, var_765);
                var_768 = wp::load(var_764);
                var_769 = wp::load(var_766);
                var_767 = wp::mul(var_768, var_769);
                var_770 = wp::sub(var_754, var_767);
                wp::assign(var_754, var_770);
                goto start_for_12;
            end_for_12:;
            // x_out[worldid, start + i] = value / L_out[worldid, factor_adr + i * size + i]       <L 3316>
            var_771 = wp::mul(var_751, var_5);
            var_772 = wp::add(var_10, var_771);
            var_773 = wp::add(var_772, var_751);
            var_774 = wp::address(var_L_out, var_0, var_773);
            var_776 = wp::load(var_774);
            var_775 = wp::div(var_754, var_776);
            var_777 = wp::add(var_3, var_751);
            wp::array_store(var_x_out, var_0, var_777, var_775);
            // i = size - 1 - reverse_i                                                           <L 3312>
            var_780 = wp::sub(var_5, var_779);
            var_781 = wp::sub(var_780, var_778);
            // value = x_out[worldid, start + i]                                                  <L 3313>
            var_782 = wp::add(var_3, var_781);
            var_783 = wp::address(var_x_out, var_0, var_782);
            var_785 = wp::load(var_783);
            var_784 = wp::copy(var_785);
            // for k in range(i + 1, size):                                                       <L 3314>
            var_787 = wp::add(var_781, var_786);
            var_789 = wp::range(var_787, var_788);
            start_for_14:;
                if (iter_cmp(var_789) == 0) goto end_for_14;
                var_790 = wp::iter_next(var_789);
                // value -= L_out[worldid, factor_adr + i * size + k] * x_out[worldid, start + k]       <L 3315>
                var_791 = wp::mul(var_781, var_5);
                var_792 = wp::add(var_10, var_791);
                var_793 = wp::add(var_792, var_790);
                var_794 = wp::address(var_L_out, var_0, var_793);
                var_795 = wp::add(var_3, var_790);
                var_796 = wp::address(var_x_out, var_0, var_795);
                var_798 = wp::load(var_794);
                var_799 = wp::load(var_796);
                var_797 = wp::mul(var_798, var_799);
                var_800 = wp::sub(var_784, var_797);
                wp::assign(var_784, var_800);
                goto start_for_14;
            end_for_14:;
            // x_out[worldid, start + i] = value / L_out[worldid, factor_adr + i * size + i]       <L 3316>
            var_801 = wp::mul(var_781, var_5);
            var_802 = wp::add(var_10, var_801);
            var_803 = wp::add(var_802, var_781);
            var_804 = wp::address(var_L_out, var_0, var_803);
            var_806 = wp::load(var_804);
            var_805 = wp::div(var_784, var_806);
            var_807 = wp::add(var_3, var_781);
            wp::array_store(var_x_out, var_0, var_807, var_805);
            // i = size - 1 - reverse_i                                                           <L 3312>
            var_810 = wp::sub(var_5, var_809);
            var_811 = wp::sub(var_810, var_808);
            // value = x_out[worldid, start + i]                                                  <L 3313>
            var_812 = wp::add(var_3, var_811);
            var_813 = wp::address(var_x_out, var_0, var_812);
            var_815 = wp::load(var_813);
            var_814 = wp::copy(var_815);
            // for k in range(i + 1, size):                                                       <L 3314>
            var_817 = wp::add(var_811, var_816);
            var_819 = wp::range(var_817, var_818);
            start_for_16:;
                if (iter_cmp(var_819) == 0) goto end_for_16;
                var_820 = wp::iter_next(var_819);
                // value -= L_out[worldid, factor_adr + i * size + k] * x_out[worldid, start + k]       <L 3315>
                var_821 = wp::mul(var_811, var_5);
                var_822 = wp::add(var_10, var_821);
                var_823 = wp::add(var_822, var_820);
                var_824 = wp::address(var_L_out, var_0, var_823);
                var_825 = wp::add(var_3, var_820);
                var_826 = wp::address(var_x_out, var_0, var_825);
                var_828 = wp::load(var_824);
                var_829 = wp::load(var_826);
                var_827 = wp::mul(var_828, var_829);
                var_830 = wp::sub(var_814, var_827);
                wp::assign(var_814, var_830);
                goto start_for_16;
            end_for_16:;
            // x_out[worldid, start + i] = value / L_out[worldid, factor_adr + i * size + i]       <L 3316>
            var_831 = wp::mul(var_811, var_5);
            var_832 = wp::add(var_10, var_831);
            var_833 = wp::add(var_832, var_811);
            var_834 = wp::address(var_L_out, var_0, var_833);
            var_836 = wp::load(var_834);
            var_835 = wp::div(var_814, var_836);
            var_837 = wp::add(var_3, var_811);
            wp::array_store(var_x_out, var_0, var_837, var_835);
            // i = size - 1 - reverse_i                                                           <L 3312>
            var_840 = wp::sub(var_5, var_839);
            var_841 = wp::sub(var_840, var_838);
            // value = x_out[worldid, start + i]                                                  <L 3313>
            var_842 = wp::add(var_3, var_841);
            var_843 = wp::address(var_x_out, var_0, var_842);
            var_845 = wp::load(var_843);
            var_844 = wp::copy(var_845);
            // for k in range(i + 1, size):                                                       <L 3314>
            var_847 = wp::add(var_841, var_846);
            var_849 = wp::range(var_847, var_848);
            start_for_18:;
                if (iter_cmp(var_849) == 0) goto end_for_18;
                var_850 = wp::iter_next(var_849);
                // value -= L_out[worldid, factor_adr + i * size + k] * x_out[worldid, start + k]       <L 3315>
                var_851 = wp::mul(var_841, var_5);
                var_852 = wp::add(var_10, var_851);
                var_853 = wp::add(var_852, var_850);
                var_854 = wp::address(var_L_out, var_0, var_853);
                var_855 = wp::add(var_3, var_850);
                var_856 = wp::address(var_x_out, var_0, var_855);
                var_858 = wp::load(var_854);
                var_859 = wp::load(var_856);
                var_857 = wp::mul(var_858, var_859);
                var_860 = wp::sub(var_844, var_857);
                wp::assign(var_844, var_860);
                goto start_for_18;
            end_for_18:;
            // x_out[worldid, start + i] = value / L_out[worldid, factor_adr + i * size + i]       <L 3316>
            var_861 = wp::mul(var_841, var_5);
            var_862 = wp::add(var_10, var_861);
            var_863 = wp::add(var_862, var_841);
            var_864 = wp::address(var_L_out, var_0, var_863);
            var_866 = wp::load(var_864);
            var_865 = wp::div(var_844, var_866);
            var_867 = wp::add(var_3, var_841);
            wp::array_store(var_x_out, var_0, var_867, var_865);
            // i = size - 1 - reverse_i                                                           <L 3312>
            var_870 = wp::sub(var_5, var_869);
            var_871 = wp::sub(var_870, var_868);
            // value = x_out[worldid, start + i]                                                  <L 3313>
            var_872 = wp::add(var_3, var_871);
            var_873 = wp::address(var_x_out, var_0, var_872);
            var_875 = wp::load(var_873);
            var_874 = wp::copy(var_875);
            // for k in range(i + 1, size):                                                       <L 3314>
            var_877 = wp::add(var_871, var_876);
            var_879 = wp::range(var_877, var_878);
            start_for_20:;
                if (iter_cmp(var_879) == 0) goto end_for_20;
                var_880 = wp::iter_next(var_879);
                // value -= L_out[worldid, factor_adr + i * size + k] * x_out[worldid, start + k]       <L 3315>
                var_881 = wp::mul(var_871, var_5);
                var_882 = wp::add(var_10, var_881);
                var_883 = wp::add(var_882, var_880);
                var_884 = wp::address(var_L_out, var_0, var_883);
                var_885 = wp::add(var_3, var_880);
                var_886 = wp::address(var_x_out, var_0, var_885);
                var_888 = wp::load(var_884);
                var_889 = wp::load(var_886);
                var_887 = wp::mul(var_888, var_889);
                var_890 = wp::sub(var_874, var_887);
                wp::assign(var_874, var_890);
                goto start_for_20;
            end_for_20:;
            // x_out[worldid, start + i] = value / L_out[worldid, factor_adr + i * size + i]       <L 3316>
            var_891 = wp::mul(var_871, var_5);
            var_892 = wp::add(var_10, var_891);
            var_893 = wp::add(var_892, var_871);
            var_894 = wp::address(var_L_out, var_0, var_893);
            var_896 = wp::load(var_894);
            var_895 = wp::div(var_874, var_896);
            var_897 = wp::add(var_3, var_871);
            wp::array_store(var_x_out, var_0, var_897, var_895);
            // i = size - 1 - reverse_i                                                           <L 3312>
            var_900 = wp::sub(var_5, var_899);
            var_901 = wp::sub(var_900, var_898);
            // value = x_out[worldid, start + i]                                                  <L 3313>
            var_902 = wp::add(var_3, var_901);
            var_903 = wp::address(var_x_out, var_0, var_902);
            var_905 = wp::load(var_903);
            var_904 = wp::copy(var_905);
            // for k in range(i + 1, size):                                                       <L 3314>
            var_907 = wp::add(var_901, var_906);
            var_909 = wp::range(var_907, var_908);
            start_for_22:;
                if (iter_cmp(var_909) == 0) goto end_for_22;
                var_910 = wp::iter_next(var_909);
                // value -= L_out[worldid, factor_adr + i * size + k] * x_out[worldid, start + k]       <L 3315>
                var_911 = wp::mul(var_901, var_5);
                var_912 = wp::add(var_10, var_911);
                var_913 = wp::add(var_912, var_910);
                var_914 = wp::address(var_L_out, var_0, var_913);
                var_915 = wp::add(var_3, var_910);
                var_916 = wp::address(var_x_out, var_0, var_915);
                var_918 = wp::load(var_914);
                var_919 = wp::load(var_916);
                var_917 = wp::mul(var_918, var_919);
                var_920 = wp::sub(var_904, var_917);
                wp::assign(var_904, var_920);
                goto start_for_22;
            end_for_22:;
            // x_out[worldid, start + i] = value / L_out[worldid, factor_adr + i * size + i]       <L 3316>
            var_921 = wp::mul(var_901, var_5);
            var_922 = wp::add(var_10, var_921);
            var_923 = wp::add(var_922, var_901);
            var_924 = wp::address(var_L_out, var_0, var_923);
            var_926 = wp::load(var_924);
            var_925 = wp::div(var_904, var_926);
            var_927 = wp::add(var_3, var_901);
            wp::array_store(var_x_out, var_0, var_927, var_925);
        }
        var_928 = wp::where(var_13, var_74, var_901);
    }
}

