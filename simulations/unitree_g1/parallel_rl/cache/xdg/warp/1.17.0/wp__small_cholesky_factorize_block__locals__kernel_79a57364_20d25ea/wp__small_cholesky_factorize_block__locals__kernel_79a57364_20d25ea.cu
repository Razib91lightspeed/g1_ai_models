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



extern "C" __global__ void _small_cholesky_factorize_block__locals__kernel_097105ab_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_qLD_block_adr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_block_dof,
    wp::array_t<wp::float32> var_D_out,
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
        const wp::int32 var_21 = 1;
        const wp::float32 var_22 = 1.0;
        wp::int32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 2;
        const wp::float32 var_29 = 1.0;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 3;
        const wp::float32 var_36 = 1.0;
        wp::int32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 4;
        const wp::float32 var_43 = 1.0;
        wp::int32 var_44;
        wp::float32* var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 5;
        const wp::float32 var_50 = 1.0;
        wp::int32 var_51;
        wp::float32* var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 0;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        wp::int32 var_59;
        const wp::int32 var_60 = 2;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32 var_63;
        wp::float32* var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        const wp::float32 var_71 = 1.0;
        wp::float32 var_72;
        const wp::int32 var_73 = 1;
        wp::int32 var_74;
        const wp::int32 var_75 = 6;
        wp::range_t var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 1;
        wp::int32 var_79;
        wp::int32 var_80;
        const wp::int32 var_81 = 2;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::int32 var_84;
        wp::float32* var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::int32 var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        const wp::int32 var_92 = 1;
        const wp::int32 var_93 = 1;
        wp::int32 var_94;
        wp::int32 var_95;
        const wp::int32 var_96 = 2;
        wp::int32 var_97;
        wp::int32 var_98;
        wp::int32 var_99;
        wp::float32* var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        const wp::int32 var_103 = 0;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::float32* var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        const wp::float32 var_116 = 1.0;
        wp::float32 var_117;
        const wp::int32 var_118 = 1;
        wp::int32 var_119;
        const wp::int32 var_120 = 6;
        wp::range_t var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        const wp::int32 var_124 = 1;
        wp::int32 var_125;
        wp::int32 var_126;
        const wp::int32 var_127 = 2;
        wp::int32 var_128;
        wp::int32 var_129;
        wp::int32 var_130;
        wp::float32* var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        const wp::int32 var_134 = 0;
        wp::int32 var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::float32* var_138;
        wp::int32 var_139;
        wp::int32 var_140;
        wp::int32 var_141;
        wp::float32* var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::int32 var_148;
        wp::int32 var_149;
        wp::int32 var_150;
        const wp::int32 var_151 = 2;
        const wp::int32 var_152 = 1;
        wp::int32 var_153;
        wp::int32 var_154;
        const wp::int32 var_155 = 2;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        wp::float32* var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        const wp::int32 var_162 = 0;
        wp::int32 var_163;
        wp::int32 var_164;
        wp::int32 var_165;
        wp::float32* var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 1;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::int32 var_174;
        wp::float32* var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        wp::float32 var_180;
        wp::int32 var_181;
        wp::int32 var_182;
        wp::int32 var_183;
        const wp::float32 var_184 = 1.0;
        wp::float32 var_185;
        const wp::int32 var_186 = 1;
        wp::int32 var_187;
        const wp::int32 var_188 = 6;
        wp::range_t var_189;
        wp::int32 var_190;
        wp::int32 var_191;
        const wp::int32 var_192 = 1;
        wp::int32 var_193;
        wp::int32 var_194;
        const wp::int32 var_195 = 2;
        wp::int32 var_196;
        wp::int32 var_197;
        wp::int32 var_198;
        wp::float32* var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        const wp::int32 var_202 = 0;
        wp::int32 var_203;
        wp::int32 var_204;
        wp::int32 var_205;
        wp::float32* var_206;
        wp::int32 var_207;
        wp::int32 var_208;
        wp::int32 var_209;
        wp::float32* var_210;
        wp::float32 var_211;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        const wp::int32 var_215 = 1;
        wp::int32 var_216;
        wp::int32 var_217;
        wp::int32 var_218;
        wp::float32* var_219;
        wp::int32 var_220;
        wp::int32 var_221;
        wp::int32 var_222;
        wp::float32* var_223;
        wp::float32 var_224;
        wp::float32 var_225;
        wp::float32 var_226;
        wp::float32 var_227;
        wp::float32 var_228;
        wp::int32 var_229;
        wp::int32 var_230;
        wp::int32 var_231;
        const wp::int32 var_232 = 3;
        const wp::int32 var_233 = 1;
        wp::int32 var_234;
        wp::int32 var_235;
        const wp::int32 var_236 = 2;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::int32 var_239;
        wp::float32* var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        const wp::int32 var_243 = 0;
        wp::int32 var_244;
        wp::int32 var_245;
        wp::int32 var_246;
        wp::float32* var_247;
        wp::float32 var_248;
        wp::float32 var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        const wp::int32 var_252 = 1;
        wp::int32 var_253;
        wp::int32 var_254;
        wp::int32 var_255;
        wp::float32* var_256;
        wp::float32 var_257;
        wp::float32 var_258;
        wp::float32 var_259;
        wp::float32 var_260;
        const wp::int32 var_261 = 2;
        wp::int32 var_262;
        wp::int32 var_263;
        wp::int32 var_264;
        wp::float32* var_265;
        wp::float32 var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        wp::float32 var_269;
        wp::float32 var_270;
        wp::int32 var_271;
        wp::int32 var_272;
        wp::int32 var_273;
        const wp::float32 var_274 = 1.0;
        wp::float32 var_275;
        const wp::int32 var_276 = 1;
        wp::int32 var_277;
        const wp::int32 var_278 = 6;
        wp::range_t var_279;
        wp::int32 var_280;
        wp::int32 var_281;
        const wp::int32 var_282 = 1;
        wp::int32 var_283;
        wp::int32 var_284;
        const wp::int32 var_285 = 2;
        wp::int32 var_286;
        wp::int32 var_287;
        wp::int32 var_288;
        wp::float32* var_289;
        wp::float32 var_290;
        wp::float32 var_291;
        const wp::int32 var_292 = 0;
        wp::int32 var_293;
        wp::int32 var_294;
        wp::int32 var_295;
        wp::float32* var_296;
        wp::int32 var_297;
        wp::int32 var_298;
        wp::int32 var_299;
        wp::float32* var_300;
        wp::float32 var_301;
        wp::float32 var_302;
        wp::float32 var_303;
        wp::float32 var_304;
        const wp::int32 var_305 = 1;
        wp::int32 var_306;
        wp::int32 var_307;
        wp::int32 var_308;
        wp::float32* var_309;
        wp::int32 var_310;
        wp::int32 var_311;
        wp::int32 var_312;
        wp::float32* var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        wp::float32 var_316;
        wp::float32 var_317;
        const wp::int32 var_318 = 2;
        wp::int32 var_319;
        wp::int32 var_320;
        wp::int32 var_321;
        wp::float32* var_322;
        wp::int32 var_323;
        wp::int32 var_324;
        wp::int32 var_325;
        wp::float32* var_326;
        wp::float32 var_327;
        wp::float32 var_328;
        wp::float32 var_329;
        wp::float32 var_330;
        wp::float32 var_331;
        wp::int32 var_332;
        wp::int32 var_333;
        wp::int32 var_334;
        const wp::int32 var_335 = 4;
        const wp::int32 var_336 = 1;
        wp::int32 var_337;
        wp::int32 var_338;
        const wp::int32 var_339 = 2;
        wp::int32 var_340;
        wp::int32 var_341;
        wp::int32 var_342;
        wp::float32* var_343;
        wp::float32 var_344;
        wp::float32 var_345;
        const wp::int32 var_346 = 0;
        wp::int32 var_347;
        wp::int32 var_348;
        wp::int32 var_349;
        wp::float32* var_350;
        wp::float32 var_351;
        wp::float32 var_352;
        wp::float32 var_353;
        wp::float32 var_354;
        const wp::int32 var_355 = 1;
        wp::int32 var_356;
        wp::int32 var_357;
        wp::int32 var_358;
        wp::float32* var_359;
        wp::float32 var_360;
        wp::float32 var_361;
        wp::float32 var_362;
        wp::float32 var_363;
        const wp::int32 var_364 = 2;
        wp::int32 var_365;
        wp::int32 var_366;
        wp::int32 var_367;
        wp::float32* var_368;
        wp::float32 var_369;
        wp::float32 var_370;
        wp::float32 var_371;
        wp::float32 var_372;
        const wp::int32 var_373 = 3;
        wp::int32 var_374;
        wp::int32 var_375;
        wp::int32 var_376;
        wp::float32* var_377;
        wp::float32 var_378;
        wp::float32 var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        wp::float32 var_382;
        wp::int32 var_383;
        wp::int32 var_384;
        wp::int32 var_385;
        const wp::float32 var_386 = 1.0;
        wp::float32 var_387;
        const wp::int32 var_388 = 1;
        wp::int32 var_389;
        const wp::int32 var_390 = 6;
        wp::range_t var_391;
        wp::int32 var_392;
        wp::int32 var_393;
        const wp::int32 var_394 = 1;
        wp::int32 var_395;
        wp::int32 var_396;
        const wp::int32 var_397 = 2;
        wp::int32 var_398;
        wp::int32 var_399;
        wp::int32 var_400;
        wp::float32* var_401;
        wp::float32 var_402;
        wp::float32 var_403;
        const wp::int32 var_404 = 0;
        wp::int32 var_405;
        wp::int32 var_406;
        wp::int32 var_407;
        wp::float32* var_408;
        wp::int32 var_409;
        wp::int32 var_410;
        wp::int32 var_411;
        wp::float32* var_412;
        wp::float32 var_413;
        wp::float32 var_414;
        wp::float32 var_415;
        wp::float32 var_416;
        const wp::int32 var_417 = 1;
        wp::int32 var_418;
        wp::int32 var_419;
        wp::int32 var_420;
        wp::float32* var_421;
        wp::int32 var_422;
        wp::int32 var_423;
        wp::int32 var_424;
        wp::float32* var_425;
        wp::float32 var_426;
        wp::float32 var_427;
        wp::float32 var_428;
        wp::float32 var_429;
        const wp::int32 var_430 = 2;
        wp::int32 var_431;
        wp::int32 var_432;
        wp::int32 var_433;
        wp::float32* var_434;
        wp::int32 var_435;
        wp::int32 var_436;
        wp::int32 var_437;
        wp::float32* var_438;
        wp::float32 var_439;
        wp::float32 var_440;
        wp::float32 var_441;
        wp::float32 var_442;
        const wp::int32 var_443 = 3;
        wp::int32 var_444;
        wp::int32 var_445;
        wp::int32 var_446;
        wp::float32* var_447;
        wp::int32 var_448;
        wp::int32 var_449;
        wp::int32 var_450;
        wp::float32* var_451;
        wp::float32 var_452;
        wp::float32 var_453;
        wp::float32 var_454;
        wp::float32 var_455;
        wp::float32 var_456;
        wp::int32 var_457;
        wp::int32 var_458;
        wp::int32 var_459;
        const wp::int32 var_460 = 5;
        const wp::int32 var_461 = 1;
        wp::int32 var_462;
        wp::int32 var_463;
        const wp::int32 var_464 = 2;
        wp::int32 var_465;
        wp::int32 var_466;
        wp::int32 var_467;
        wp::float32* var_468;
        wp::float32 var_469;
        wp::float32 var_470;
        const wp::int32 var_471 = 0;
        wp::int32 var_472;
        wp::int32 var_473;
        wp::int32 var_474;
        wp::float32* var_475;
        wp::float32 var_476;
        wp::float32 var_477;
        wp::float32 var_478;
        wp::float32 var_479;
        const wp::int32 var_480 = 1;
        wp::int32 var_481;
        wp::int32 var_482;
        wp::int32 var_483;
        wp::float32* var_484;
        wp::float32 var_485;
        wp::float32 var_486;
        wp::float32 var_487;
        wp::float32 var_488;
        const wp::int32 var_489 = 2;
        wp::int32 var_490;
        wp::int32 var_491;
        wp::int32 var_492;
        wp::float32* var_493;
        wp::float32 var_494;
        wp::float32 var_495;
        wp::float32 var_496;
        wp::float32 var_497;
        const wp::int32 var_498 = 3;
        wp::int32 var_499;
        wp::int32 var_500;
        wp::int32 var_501;
        wp::float32* var_502;
        wp::float32 var_503;
        wp::float32 var_504;
        wp::float32 var_505;
        wp::float32 var_506;
        const wp::int32 var_507 = 4;
        wp::int32 var_508;
        wp::int32 var_509;
        wp::int32 var_510;
        wp::float32* var_511;
        wp::float32 var_512;
        wp::float32 var_513;
        wp::float32 var_514;
        wp::float32 var_515;
        wp::float32 var_516;
        wp::int32 var_517;
        wp::int32 var_518;
        wp::int32 var_519;
        const wp::float32 var_520 = 1.0;
        wp::float32 var_521;
        const wp::int32 var_522 = 1;
        wp::int32 var_523;
        const wp::int32 var_524 = 6;
        wp::range_t var_525;
        wp::int32 var_526;
        wp::int32 var_527;
        const wp::int32 var_528 = 1;
        wp::int32 var_529;
        wp::int32 var_530;
        const wp::int32 var_531 = 2;
        wp::int32 var_532;
        wp::int32 var_533;
        wp::int32 var_534;
        wp::float32* var_535;
        wp::float32 var_536;
        wp::float32 var_537;
        const wp::int32 var_538 = 0;
        wp::int32 var_539;
        wp::int32 var_540;
        wp::int32 var_541;
        wp::float32* var_542;
        wp::int32 var_543;
        wp::int32 var_544;
        wp::int32 var_545;
        wp::float32* var_546;
        wp::float32 var_547;
        wp::float32 var_548;
        wp::float32 var_549;
        wp::float32 var_550;
        const wp::int32 var_551 = 1;
        wp::int32 var_552;
        wp::int32 var_553;
        wp::int32 var_554;
        wp::float32* var_555;
        wp::int32 var_556;
        wp::int32 var_557;
        wp::int32 var_558;
        wp::float32* var_559;
        wp::float32 var_560;
        wp::float32 var_561;
        wp::float32 var_562;
        wp::float32 var_563;
        const wp::int32 var_564 = 2;
        wp::int32 var_565;
        wp::int32 var_566;
        wp::int32 var_567;
        wp::float32* var_568;
        wp::int32 var_569;
        wp::int32 var_570;
        wp::int32 var_571;
        wp::float32* var_572;
        wp::float32 var_573;
        wp::float32 var_574;
        wp::float32 var_575;
        wp::float32 var_576;
        const wp::int32 var_577 = 3;
        wp::int32 var_578;
        wp::int32 var_579;
        wp::int32 var_580;
        wp::float32* var_581;
        wp::int32 var_582;
        wp::int32 var_583;
        wp::int32 var_584;
        wp::float32* var_585;
        wp::float32 var_586;
        wp::float32 var_587;
        wp::float32 var_588;
        wp::float32 var_589;
        const wp::int32 var_590 = 4;
        wp::int32 var_591;
        wp::int32 var_592;
        wp::int32 var_593;
        wp::float32* var_594;
        wp::int32 var_595;
        wp::int32 var_596;
        wp::int32 var_597;
        wp::float32* var_598;
        wp::float32 var_599;
        wp::float32 var_600;
        wp::float32 var_601;
        wp::float32 var_602;
        wp::float32 var_603;
        wp::int32 var_604;
        wp::int32 var_605;
        wp::int32 var_606;
        wp::int32 var_607;
        //---------
        // forward
        // def kernel(                                                                            <L 1238>
        // worldid, blk = wp.tid()                                                                <L 1250>
        builtin_tid2d(var_0, var_1);
        // start = block_dof[blk]                                                                 <L 1251>
        var_2 = wp::address(var_block_dof, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // size = wp.static(block_size)                                                           <L 1252>
        // matrix_adr = M_rowadr[start]                                                           <L 1253>
        var_6 = wp::address(var_M_rowadr, var_3);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // factor_adr = qLD_block_adr[start]                                                      <L 1255>
        var_9 = wp::address(var_qLD_block_adr, var_3);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if factor_adr == Q_LD_BLOCK_COMPACT:                                                   <L 1256>
        var_13 = (var_10 == var_12);
        if (var_13) {
            // for i in range(wp.static(block_size)):                                             <L 1257>
            // D_out[worldid, start + i] = 1.0 / M_in[worldid, matrix_adr + i]                    <L 1258>
            var_16 = wp::add(var_7, var_14);
            var_17 = wp::address(var_M_in, var_0, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::div(var_15, var_19);
            var_20 = wp::add(var_3, var_14);
            wp::array_store(var_D_out, var_0, var_20, var_18);
            var_23 = wp::add(var_7, var_21);
            var_24 = wp::address(var_M_in, var_0, var_23);
            var_26 = wp::load(var_24);
            var_25 = wp::div(var_22, var_26);
            var_27 = wp::add(var_3, var_21);
            wp::array_store(var_D_out, var_0, var_27, var_25);
            var_30 = wp::add(var_7, var_28);
            var_31 = wp::address(var_M_in, var_0, var_30);
            var_33 = wp::load(var_31);
            var_32 = wp::div(var_29, var_33);
            var_34 = wp::add(var_3, var_28);
            wp::array_store(var_D_out, var_0, var_34, var_32);
            var_37 = wp::add(var_7, var_35);
            var_38 = wp::address(var_M_in, var_0, var_37);
            var_40 = wp::load(var_38);
            var_39 = wp::div(var_36, var_40);
            var_41 = wp::add(var_3, var_35);
            wp::array_store(var_D_out, var_0, var_41, var_39);
            var_44 = wp::add(var_7, var_42);
            var_45 = wp::address(var_M_in, var_0, var_44);
            var_47 = wp::load(var_45);
            var_46 = wp::div(var_43, var_47);
            var_48 = wp::add(var_3, var_42);
            wp::array_store(var_D_out, var_0, var_48, var_46);
            var_51 = wp::add(var_7, var_49);
            var_52 = wp::address(var_M_in, var_0, var_51);
            var_54 = wp::load(var_52);
            var_53 = wp::div(var_50, var_54);
            var_55 = wp::add(var_3, var_49);
            wp::array_store(var_D_out, var_0, var_55, var_53);
        }
        if (!var_13) {
            // for i in range(wp.static(block_size)):                                             <L 1260>
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_58 = wp::add(var_56, var_57);
            var_59 = wp::mul(var_56, var_58);
            var_61 = wp::floordiv(var_59, var_60);
            var_62 = wp::add(var_7, var_61);
            var_63 = wp::add(var_62, var_56);
            var_64 = wp::address(var_M_in, var_0, var_63);
            var_66 = wp::load(var_64);
            var_65 = wp::copy(var_66);
            // for k in range(i):                                                                 <L 1262>
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_67 = wp::sqrt(var_65);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_68 = wp::mul(var_56, var_5);
            var_69 = wp::add(var_10, var_68);
            var_70 = wp::add(var_69, var_56);
            wp::array_store(var_L_out, var_0, var_70, var_67);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_72 = wp::div(var_71, var_67);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_74 = wp::add(var_56, var_73);
            var_76 = wp::range(var_74, var_75);
            start_for_0:;
                if (iter_cmp(var_76) == 0) goto end_for_0;
                var_77 = wp::iter_next(var_76);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_79 = wp::add(var_77, var_78);
                var_80 = wp::mul(var_77, var_79);
                var_82 = wp::floordiv(var_80, var_81);
                var_83 = wp::add(var_7, var_82);
                var_84 = wp::add(var_83, var_56);
                var_85 = wp::address(var_M_in, var_0, var_84);
                var_87 = wp::load(var_85);
                var_86 = wp::copy(var_87);
                // for k in range(i):                                                             <L 1272>
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_88 = wp::mul(var_86, var_72);
                var_89 = wp::mul(var_56, var_5);
                var_90 = wp::add(var_10, var_89);
                var_91 = wp::add(var_90, var_77);
                wp::array_store(var_L_out, var_0, var_91, var_88);
                wp::assign(var_65, var_86);
                goto start_for_0;
            end_for_0:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_94 = wp::add(var_92, var_93);
            var_95 = wp::mul(var_92, var_94);
            var_97 = wp::floordiv(var_95, var_96);
            var_98 = wp::add(var_7, var_97);
            var_99 = wp::add(var_98, var_92);
            var_100 = wp::address(var_M_in, var_0, var_99);
            var_102 = wp::load(var_100);
            var_101 = wp::copy(var_102);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_104 = wp::mul(var_103, var_5);
            var_105 = wp::add(var_10, var_104);
            var_106 = wp::add(var_105, var_92);
            var_107 = wp::address(var_L_out, var_0, var_106);
            var_109 = wp::load(var_107);
            var_108 = wp::copy(var_109);
            // value -= factor * factor                                                           <L 1264>
            var_110 = wp::mul(var_108, var_108);
            var_111 = wp::sub(var_101, var_110);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_112 = wp::sqrt(var_111);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_113 = wp::mul(var_92, var_5);
            var_114 = wp::add(var_10, var_113);
            var_115 = wp::add(var_114, var_92);
            wp::array_store(var_L_out, var_0, var_115, var_112);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_117 = wp::div(var_116, var_112);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_119 = wp::add(var_92, var_118);
            var_121 = wp::range(var_119, var_120);
            var_122 = wp::copy(var_103);
            start_for_2:;
                if (iter_cmp(var_121) == 0) goto end_for_2;
                var_123 = wp::iter_next(var_121);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_125 = wp::add(var_123, var_124);
                var_126 = wp::mul(var_123, var_125);
                var_128 = wp::floordiv(var_126, var_127);
                var_129 = wp::add(var_7, var_128);
                var_130 = wp::add(var_129, var_92);
                var_131 = wp::address(var_M_in, var_0, var_130);
                var_133 = wp::load(var_131);
                var_132 = wp::copy(var_133);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_135 = wp::mul(var_134, var_5);
                var_136 = wp::add(var_10, var_135);
                var_137 = wp::add(var_136, var_92);
                var_138 = wp::address(var_L_out, var_0, var_137);
                var_139 = wp::mul(var_134, var_5);
                var_140 = wp::add(var_10, var_139);
                var_141 = wp::add(var_140, var_123);
                var_142 = wp::address(var_L_out, var_0, var_141);
                var_144 = wp::load(var_138);
                var_145 = wp::load(var_142);
                var_143 = wp::mul(var_144, var_145);
                var_146 = wp::sub(var_132, var_143);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_147 = wp::mul(var_146, var_117);
                var_148 = wp::mul(var_92, var_5);
                var_149 = wp::add(var_10, var_148);
                var_150 = wp::add(var_149, var_123);
                wp::array_store(var_L_out, var_0, var_150, var_147);
                wp::assign(var_111, var_146);
                wp::assign(var_122, var_134);
                goto start_for_2;
            end_for_2:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_153 = wp::add(var_151, var_152);
            var_154 = wp::mul(var_151, var_153);
            var_156 = wp::floordiv(var_154, var_155);
            var_157 = wp::add(var_7, var_156);
            var_158 = wp::add(var_157, var_151);
            var_159 = wp::address(var_M_in, var_0, var_158);
            var_161 = wp::load(var_159);
            var_160 = wp::copy(var_161);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_163 = wp::mul(var_162, var_5);
            var_164 = wp::add(var_10, var_163);
            var_165 = wp::add(var_164, var_151);
            var_166 = wp::address(var_L_out, var_0, var_165);
            var_168 = wp::load(var_166);
            var_167 = wp::copy(var_168);
            // value -= factor * factor                                                           <L 1264>
            var_169 = wp::mul(var_167, var_167);
            var_170 = wp::sub(var_160, var_169);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_172 = wp::mul(var_171, var_5);
            var_173 = wp::add(var_10, var_172);
            var_174 = wp::add(var_173, var_151);
            var_175 = wp::address(var_L_out, var_0, var_174);
            var_177 = wp::load(var_175);
            var_176 = wp::copy(var_177);
            // value -= factor * factor                                                           <L 1264>
            var_178 = wp::mul(var_176, var_176);
            var_179 = wp::sub(var_170, var_178);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_180 = wp::sqrt(var_179);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_181 = wp::mul(var_151, var_5);
            var_182 = wp::add(var_10, var_181);
            var_183 = wp::add(var_182, var_151);
            wp::array_store(var_L_out, var_0, var_183, var_180);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_185 = wp::div(var_184, var_180);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_187 = wp::add(var_151, var_186);
            var_189 = wp::range(var_187, var_188);
            var_190 = wp::copy(var_171);
            start_for_4:;
                if (iter_cmp(var_189) == 0) goto end_for_4;
                var_191 = wp::iter_next(var_189);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_193 = wp::add(var_191, var_192);
                var_194 = wp::mul(var_191, var_193);
                var_196 = wp::floordiv(var_194, var_195);
                var_197 = wp::add(var_7, var_196);
                var_198 = wp::add(var_197, var_151);
                var_199 = wp::address(var_M_in, var_0, var_198);
                var_201 = wp::load(var_199);
                var_200 = wp::copy(var_201);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_203 = wp::mul(var_202, var_5);
                var_204 = wp::add(var_10, var_203);
                var_205 = wp::add(var_204, var_151);
                var_206 = wp::address(var_L_out, var_0, var_205);
                var_207 = wp::mul(var_202, var_5);
                var_208 = wp::add(var_10, var_207);
                var_209 = wp::add(var_208, var_191);
                var_210 = wp::address(var_L_out, var_0, var_209);
                var_212 = wp::load(var_206);
                var_213 = wp::load(var_210);
                var_211 = wp::mul(var_212, var_213);
                var_214 = wp::sub(var_200, var_211);
                var_216 = wp::mul(var_215, var_5);
                var_217 = wp::add(var_10, var_216);
                var_218 = wp::add(var_217, var_151);
                var_219 = wp::address(var_L_out, var_0, var_218);
                var_220 = wp::mul(var_215, var_5);
                var_221 = wp::add(var_10, var_220);
                var_222 = wp::add(var_221, var_191);
                var_223 = wp::address(var_L_out, var_0, var_222);
                var_225 = wp::load(var_219);
                var_226 = wp::load(var_223);
                var_224 = wp::mul(var_225, var_226);
                var_227 = wp::sub(var_214, var_224);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_228 = wp::mul(var_227, var_185);
                var_229 = wp::mul(var_151, var_5);
                var_230 = wp::add(var_10, var_229);
                var_231 = wp::add(var_230, var_191);
                wp::array_store(var_L_out, var_0, var_231, var_228);
                wp::assign(var_179, var_227);
                wp::assign(var_190, var_215);
                goto start_for_4;
            end_for_4:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_234 = wp::add(var_232, var_233);
            var_235 = wp::mul(var_232, var_234);
            var_237 = wp::floordiv(var_235, var_236);
            var_238 = wp::add(var_7, var_237);
            var_239 = wp::add(var_238, var_232);
            var_240 = wp::address(var_M_in, var_0, var_239);
            var_242 = wp::load(var_240);
            var_241 = wp::copy(var_242);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_244 = wp::mul(var_243, var_5);
            var_245 = wp::add(var_10, var_244);
            var_246 = wp::add(var_245, var_232);
            var_247 = wp::address(var_L_out, var_0, var_246);
            var_249 = wp::load(var_247);
            var_248 = wp::copy(var_249);
            // value -= factor * factor                                                           <L 1264>
            var_250 = wp::mul(var_248, var_248);
            var_251 = wp::sub(var_241, var_250);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_253 = wp::mul(var_252, var_5);
            var_254 = wp::add(var_10, var_253);
            var_255 = wp::add(var_254, var_232);
            var_256 = wp::address(var_L_out, var_0, var_255);
            var_258 = wp::load(var_256);
            var_257 = wp::copy(var_258);
            // value -= factor * factor                                                           <L 1264>
            var_259 = wp::mul(var_257, var_257);
            var_260 = wp::sub(var_251, var_259);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_262 = wp::mul(var_261, var_5);
            var_263 = wp::add(var_10, var_262);
            var_264 = wp::add(var_263, var_232);
            var_265 = wp::address(var_L_out, var_0, var_264);
            var_267 = wp::load(var_265);
            var_266 = wp::copy(var_267);
            // value -= factor * factor                                                           <L 1264>
            var_268 = wp::mul(var_266, var_266);
            var_269 = wp::sub(var_260, var_268);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_270 = wp::sqrt(var_269);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_271 = wp::mul(var_232, var_5);
            var_272 = wp::add(var_10, var_271);
            var_273 = wp::add(var_272, var_232);
            wp::array_store(var_L_out, var_0, var_273, var_270);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_275 = wp::div(var_274, var_270);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_277 = wp::add(var_232, var_276);
            var_279 = wp::range(var_277, var_278);
            var_280 = wp::copy(var_261);
            start_for_6:;
                if (iter_cmp(var_279) == 0) goto end_for_6;
                var_281 = wp::iter_next(var_279);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_283 = wp::add(var_281, var_282);
                var_284 = wp::mul(var_281, var_283);
                var_286 = wp::floordiv(var_284, var_285);
                var_287 = wp::add(var_7, var_286);
                var_288 = wp::add(var_287, var_232);
                var_289 = wp::address(var_M_in, var_0, var_288);
                var_291 = wp::load(var_289);
                var_290 = wp::copy(var_291);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_293 = wp::mul(var_292, var_5);
                var_294 = wp::add(var_10, var_293);
                var_295 = wp::add(var_294, var_232);
                var_296 = wp::address(var_L_out, var_0, var_295);
                var_297 = wp::mul(var_292, var_5);
                var_298 = wp::add(var_10, var_297);
                var_299 = wp::add(var_298, var_281);
                var_300 = wp::address(var_L_out, var_0, var_299);
                var_302 = wp::load(var_296);
                var_303 = wp::load(var_300);
                var_301 = wp::mul(var_302, var_303);
                var_304 = wp::sub(var_290, var_301);
                var_306 = wp::mul(var_305, var_5);
                var_307 = wp::add(var_10, var_306);
                var_308 = wp::add(var_307, var_232);
                var_309 = wp::address(var_L_out, var_0, var_308);
                var_310 = wp::mul(var_305, var_5);
                var_311 = wp::add(var_10, var_310);
                var_312 = wp::add(var_311, var_281);
                var_313 = wp::address(var_L_out, var_0, var_312);
                var_315 = wp::load(var_309);
                var_316 = wp::load(var_313);
                var_314 = wp::mul(var_315, var_316);
                var_317 = wp::sub(var_304, var_314);
                var_319 = wp::mul(var_318, var_5);
                var_320 = wp::add(var_10, var_319);
                var_321 = wp::add(var_320, var_232);
                var_322 = wp::address(var_L_out, var_0, var_321);
                var_323 = wp::mul(var_318, var_5);
                var_324 = wp::add(var_10, var_323);
                var_325 = wp::add(var_324, var_281);
                var_326 = wp::address(var_L_out, var_0, var_325);
                var_328 = wp::load(var_322);
                var_329 = wp::load(var_326);
                var_327 = wp::mul(var_328, var_329);
                var_330 = wp::sub(var_317, var_327);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_331 = wp::mul(var_330, var_275);
                var_332 = wp::mul(var_232, var_5);
                var_333 = wp::add(var_10, var_332);
                var_334 = wp::add(var_333, var_281);
                wp::array_store(var_L_out, var_0, var_334, var_331);
                wp::assign(var_269, var_330);
                wp::assign(var_280, var_318);
                goto start_for_6;
            end_for_6:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_337 = wp::add(var_335, var_336);
            var_338 = wp::mul(var_335, var_337);
            var_340 = wp::floordiv(var_338, var_339);
            var_341 = wp::add(var_7, var_340);
            var_342 = wp::add(var_341, var_335);
            var_343 = wp::address(var_M_in, var_0, var_342);
            var_345 = wp::load(var_343);
            var_344 = wp::copy(var_345);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_347 = wp::mul(var_346, var_5);
            var_348 = wp::add(var_10, var_347);
            var_349 = wp::add(var_348, var_335);
            var_350 = wp::address(var_L_out, var_0, var_349);
            var_352 = wp::load(var_350);
            var_351 = wp::copy(var_352);
            // value -= factor * factor                                                           <L 1264>
            var_353 = wp::mul(var_351, var_351);
            var_354 = wp::sub(var_344, var_353);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_356 = wp::mul(var_355, var_5);
            var_357 = wp::add(var_10, var_356);
            var_358 = wp::add(var_357, var_335);
            var_359 = wp::address(var_L_out, var_0, var_358);
            var_361 = wp::load(var_359);
            var_360 = wp::copy(var_361);
            // value -= factor * factor                                                           <L 1264>
            var_362 = wp::mul(var_360, var_360);
            var_363 = wp::sub(var_354, var_362);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_365 = wp::mul(var_364, var_5);
            var_366 = wp::add(var_10, var_365);
            var_367 = wp::add(var_366, var_335);
            var_368 = wp::address(var_L_out, var_0, var_367);
            var_370 = wp::load(var_368);
            var_369 = wp::copy(var_370);
            // value -= factor * factor                                                           <L 1264>
            var_371 = wp::mul(var_369, var_369);
            var_372 = wp::sub(var_363, var_371);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_374 = wp::mul(var_373, var_5);
            var_375 = wp::add(var_10, var_374);
            var_376 = wp::add(var_375, var_335);
            var_377 = wp::address(var_L_out, var_0, var_376);
            var_379 = wp::load(var_377);
            var_378 = wp::copy(var_379);
            // value -= factor * factor                                                           <L 1264>
            var_380 = wp::mul(var_378, var_378);
            var_381 = wp::sub(var_372, var_380);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_382 = wp::sqrt(var_381);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_383 = wp::mul(var_335, var_5);
            var_384 = wp::add(var_10, var_383);
            var_385 = wp::add(var_384, var_335);
            wp::array_store(var_L_out, var_0, var_385, var_382);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_387 = wp::div(var_386, var_382);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_389 = wp::add(var_335, var_388);
            var_391 = wp::range(var_389, var_390);
            var_392 = wp::copy(var_373);
            start_for_8:;
                if (iter_cmp(var_391) == 0) goto end_for_8;
                var_393 = wp::iter_next(var_391);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_395 = wp::add(var_393, var_394);
                var_396 = wp::mul(var_393, var_395);
                var_398 = wp::floordiv(var_396, var_397);
                var_399 = wp::add(var_7, var_398);
                var_400 = wp::add(var_399, var_335);
                var_401 = wp::address(var_M_in, var_0, var_400);
                var_403 = wp::load(var_401);
                var_402 = wp::copy(var_403);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_405 = wp::mul(var_404, var_5);
                var_406 = wp::add(var_10, var_405);
                var_407 = wp::add(var_406, var_335);
                var_408 = wp::address(var_L_out, var_0, var_407);
                var_409 = wp::mul(var_404, var_5);
                var_410 = wp::add(var_10, var_409);
                var_411 = wp::add(var_410, var_393);
                var_412 = wp::address(var_L_out, var_0, var_411);
                var_414 = wp::load(var_408);
                var_415 = wp::load(var_412);
                var_413 = wp::mul(var_414, var_415);
                var_416 = wp::sub(var_402, var_413);
                var_418 = wp::mul(var_417, var_5);
                var_419 = wp::add(var_10, var_418);
                var_420 = wp::add(var_419, var_335);
                var_421 = wp::address(var_L_out, var_0, var_420);
                var_422 = wp::mul(var_417, var_5);
                var_423 = wp::add(var_10, var_422);
                var_424 = wp::add(var_423, var_393);
                var_425 = wp::address(var_L_out, var_0, var_424);
                var_427 = wp::load(var_421);
                var_428 = wp::load(var_425);
                var_426 = wp::mul(var_427, var_428);
                var_429 = wp::sub(var_416, var_426);
                var_431 = wp::mul(var_430, var_5);
                var_432 = wp::add(var_10, var_431);
                var_433 = wp::add(var_432, var_335);
                var_434 = wp::address(var_L_out, var_0, var_433);
                var_435 = wp::mul(var_430, var_5);
                var_436 = wp::add(var_10, var_435);
                var_437 = wp::add(var_436, var_393);
                var_438 = wp::address(var_L_out, var_0, var_437);
                var_440 = wp::load(var_434);
                var_441 = wp::load(var_438);
                var_439 = wp::mul(var_440, var_441);
                var_442 = wp::sub(var_429, var_439);
                var_444 = wp::mul(var_443, var_5);
                var_445 = wp::add(var_10, var_444);
                var_446 = wp::add(var_445, var_335);
                var_447 = wp::address(var_L_out, var_0, var_446);
                var_448 = wp::mul(var_443, var_5);
                var_449 = wp::add(var_10, var_448);
                var_450 = wp::add(var_449, var_393);
                var_451 = wp::address(var_L_out, var_0, var_450);
                var_453 = wp::load(var_447);
                var_454 = wp::load(var_451);
                var_452 = wp::mul(var_453, var_454);
                var_455 = wp::sub(var_442, var_452);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_456 = wp::mul(var_455, var_387);
                var_457 = wp::mul(var_335, var_5);
                var_458 = wp::add(var_10, var_457);
                var_459 = wp::add(var_458, var_393);
                wp::array_store(var_L_out, var_0, var_459, var_456);
                wp::assign(var_381, var_455);
                wp::assign(var_392, var_443);
                goto start_for_8;
            end_for_8:;
            // value = M_in[worldid, matrix_adr + i * (i + 1) // 2 + i]                           <L 1261>
            var_462 = wp::add(var_460, var_461);
            var_463 = wp::mul(var_460, var_462);
            var_465 = wp::floordiv(var_463, var_464);
            var_466 = wp::add(var_7, var_465);
            var_467 = wp::add(var_466, var_460);
            var_468 = wp::address(var_M_in, var_0, var_467);
            var_470 = wp::load(var_468);
            var_469 = wp::copy(var_470);
            // for k in range(i):                                                                 <L 1262>
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_472 = wp::mul(var_471, var_5);
            var_473 = wp::add(var_10, var_472);
            var_474 = wp::add(var_473, var_460);
            var_475 = wp::address(var_L_out, var_0, var_474);
            var_477 = wp::load(var_475);
            var_476 = wp::copy(var_477);
            // value -= factor * factor                                                           <L 1264>
            var_478 = wp::mul(var_476, var_476);
            var_479 = wp::sub(var_469, var_478);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_481 = wp::mul(var_480, var_5);
            var_482 = wp::add(var_10, var_481);
            var_483 = wp::add(var_482, var_460);
            var_484 = wp::address(var_L_out, var_0, var_483);
            var_486 = wp::load(var_484);
            var_485 = wp::copy(var_486);
            // value -= factor * factor                                                           <L 1264>
            var_487 = wp::mul(var_485, var_485);
            var_488 = wp::sub(var_479, var_487);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_490 = wp::mul(var_489, var_5);
            var_491 = wp::add(var_10, var_490);
            var_492 = wp::add(var_491, var_460);
            var_493 = wp::address(var_L_out, var_0, var_492);
            var_495 = wp::load(var_493);
            var_494 = wp::copy(var_495);
            // value -= factor * factor                                                           <L 1264>
            var_496 = wp::mul(var_494, var_494);
            var_497 = wp::sub(var_488, var_496);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_499 = wp::mul(var_498, var_5);
            var_500 = wp::add(var_10, var_499);
            var_501 = wp::add(var_500, var_460);
            var_502 = wp::address(var_L_out, var_0, var_501);
            var_504 = wp::load(var_502);
            var_503 = wp::copy(var_504);
            // value -= factor * factor                                                           <L 1264>
            var_505 = wp::mul(var_503, var_503);
            var_506 = wp::sub(var_497, var_505);
            // factor = L_out[worldid, factor_adr + k * size + i]                                 <L 1263>
            var_508 = wp::mul(var_507, var_5);
            var_509 = wp::add(var_10, var_508);
            var_510 = wp::add(var_509, var_460);
            var_511 = wp::address(var_L_out, var_0, var_510);
            var_513 = wp::load(var_511);
            var_512 = wp::copy(var_513);
            // value -= factor * factor                                                           <L 1264>
            var_514 = wp::mul(var_512, var_512);
            var_515 = wp::sub(var_506, var_514);
            // diagonal_value = wp.sqrt(value)                                                    <L 1266>
            var_516 = wp::sqrt(var_515);
            // L_out[worldid, factor_adr + i * size + i] = diagonal_value                         <L 1267>
            var_517 = wp::mul(var_460, var_5);
            var_518 = wp::add(var_10, var_517);
            var_519 = wp::add(var_518, var_460);
            wp::array_store(var_L_out, var_0, var_519, var_516);
            // diagonal_inv = 1.0 / diagonal_value                                                <L 1268>
            var_521 = wp::div(var_520, var_516);
            // for j in range(i + 1, size):                                                       <L 1270>
            var_523 = wp::add(var_460, var_522);
            var_525 = wp::range(var_523, var_524);
            var_526 = wp::copy(var_507);
            start_for_10:;
                if (iter_cmp(var_525) == 0) goto end_for_10;
                var_527 = wp::iter_next(var_525);
                // value = M_in[worldid, matrix_adr + j * (j + 1) // 2 + i]                       <L 1271>
                var_529 = wp::add(var_527, var_528);
                var_530 = wp::mul(var_527, var_529);
                var_532 = wp::floordiv(var_530, var_531);
                var_533 = wp::add(var_7, var_532);
                var_534 = wp::add(var_533, var_460);
                var_535 = wp::address(var_M_in, var_0, var_534);
                var_537 = wp::load(var_535);
                var_536 = wp::copy(var_537);
                // for k in range(i):                                                             <L 1272>
                // value -= L_out[worldid, factor_adr + k * size + i] * L_out[worldid, factor_adr + k * size + j]       <L 1273>
                var_539 = wp::mul(var_538, var_5);
                var_540 = wp::add(var_10, var_539);
                var_541 = wp::add(var_540, var_460);
                var_542 = wp::address(var_L_out, var_0, var_541);
                var_543 = wp::mul(var_538, var_5);
                var_544 = wp::add(var_10, var_543);
                var_545 = wp::add(var_544, var_527);
                var_546 = wp::address(var_L_out, var_0, var_545);
                var_548 = wp::load(var_542);
                var_549 = wp::load(var_546);
                var_547 = wp::mul(var_548, var_549);
                var_550 = wp::sub(var_536, var_547);
                var_552 = wp::mul(var_551, var_5);
                var_553 = wp::add(var_10, var_552);
                var_554 = wp::add(var_553, var_460);
                var_555 = wp::address(var_L_out, var_0, var_554);
                var_556 = wp::mul(var_551, var_5);
                var_557 = wp::add(var_10, var_556);
                var_558 = wp::add(var_557, var_527);
                var_559 = wp::address(var_L_out, var_0, var_558);
                var_561 = wp::load(var_555);
                var_562 = wp::load(var_559);
                var_560 = wp::mul(var_561, var_562);
                var_563 = wp::sub(var_550, var_560);
                var_565 = wp::mul(var_564, var_5);
                var_566 = wp::add(var_10, var_565);
                var_567 = wp::add(var_566, var_460);
                var_568 = wp::address(var_L_out, var_0, var_567);
                var_569 = wp::mul(var_564, var_5);
                var_570 = wp::add(var_10, var_569);
                var_571 = wp::add(var_570, var_527);
                var_572 = wp::address(var_L_out, var_0, var_571);
                var_574 = wp::load(var_568);
                var_575 = wp::load(var_572);
                var_573 = wp::mul(var_574, var_575);
                var_576 = wp::sub(var_563, var_573);
                var_578 = wp::mul(var_577, var_5);
                var_579 = wp::add(var_10, var_578);
                var_580 = wp::add(var_579, var_460);
                var_581 = wp::address(var_L_out, var_0, var_580);
                var_582 = wp::mul(var_577, var_5);
                var_583 = wp::add(var_10, var_582);
                var_584 = wp::add(var_583, var_527);
                var_585 = wp::address(var_L_out, var_0, var_584);
                var_587 = wp::load(var_581);
                var_588 = wp::load(var_585);
                var_586 = wp::mul(var_587, var_588);
                var_589 = wp::sub(var_576, var_586);
                var_591 = wp::mul(var_590, var_5);
                var_592 = wp::add(var_10, var_591);
                var_593 = wp::add(var_592, var_460);
                var_594 = wp::address(var_L_out, var_0, var_593);
                var_595 = wp::mul(var_590, var_5);
                var_596 = wp::add(var_10, var_595);
                var_597 = wp::add(var_596, var_527);
                var_598 = wp::address(var_L_out, var_0, var_597);
                var_600 = wp::load(var_594);
                var_601 = wp::load(var_598);
                var_599 = wp::mul(var_600, var_601);
                var_602 = wp::sub(var_589, var_599);
                // L_out[worldid, factor_adr + i * size + j] = value * diagonal_inv               <L 1274>
                var_603 = wp::mul(var_602, var_521);
                var_604 = wp::mul(var_460, var_5);
                var_605 = wp::add(var_10, var_604);
                var_606 = wp::add(var_605, var_527);
                wp::array_store(var_L_out, var_0, var_606, var_603);
                wp::assign(var_515, var_602);
                wp::assign(var_526, var_590);
                goto start_for_10;
            end_for_10:;
        }
        var_607 = wp::where(var_13, var_49, var_460);
    }
}

