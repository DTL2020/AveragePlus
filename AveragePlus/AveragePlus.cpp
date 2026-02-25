#include "AveragePlus.h"

#ifdef INTEL_INTRINSICS
#include "average_avx.h"
#include "average_avx2.h"
#endif

#ifdef INTEL_INTRINSICS
#include "emmintrin.h"
#endif

#include <cstring>

#if defined(_WIN32) && !defined(INTEL_INTRINSICS)
#error Forgot to set INTEL_INTRINSICS? Comment out this line if not
#endif

template<int minimum, int maximum>
static AVS_FORCEINLINE int static_clip(float val) {
    if (val > maximum) {
        return maximum;
    }
    if (val < minimum) {
        return minimum;
    }
    return (int)val;
}

template<typename pixel_t, int bits_per_pixel>
static AVS_FORCEINLINE void weighted_average_c(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height) {
  // width is rowsize
  const int max_pixel_value = (sizeof(pixel_t) == 1) ? 255 : ((1 << bits_per_pixel) - 1);

  width /= sizeof(pixel_t);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float acc = 0;
      for (int i = 0; i < frames_count; ++i) {
        acc += reinterpret_cast<const pixel_t *>(src_pointers[i])[x] * weights[i];
      }
      if (sizeof(pixel_t) == 4)
        reinterpret_cast<float *>(dstp)[x] = acc;
      else
        reinterpret_cast<pixel_t *>(dstp)[x] = (pixel_t)(static_clip<0, max_pixel_value>(acc));
    }

    for (int i = 0; i < frames_count; ++i) {
      src_pointers[i] += src_pitches[i];
    }
    dstp += dst_pitch;
  }
}

#ifdef INTEL_INTRINSICS
// fake _mm_packus_epi32 (orig is SSE4.1 only)
static AVS_FORCEINLINE __m128i _MM_PACKUS_EPI32(__m128i a, __m128i b)
{
  a = _mm_slli_epi32(a, 16);
  a = _mm_srai_epi32(a, 16);
  b = _mm_slli_epi32(b, 16);
  b = _mm_srai_epi32(b, 16);
  a = _mm_packs_epi32(a, b);
  return a;
}

template<typename pixel_t, int bits_per_pixel>
static inline void weighted_average_sse2(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height) {
    // width is row_size
    int mod_width;
    if(sizeof(pixel_t) == 1)
      mod_width = width / 8 * 8;
    else
      mod_width = width / 16 * 16;

    const int sse_size = (sizeof(pixel_t) == 1) ? 8 : 16;

    const int max_pixel_value = (sizeof(pixel_t) == 1) ? 255 : ((1 << bits_per_pixel) - 1);
    __m128i pixel_limit;
    if (sizeof(pixel_t) == 2 && bits_per_pixel < 16)
      pixel_limit = _mm_set1_epi16((int16_t)max_pixel_value);

    __m128i zero = _mm_setzero_si128();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < mod_width; x += sse_size) {
            __m128 acc_lo = _mm_setzero_ps();
            __m128 acc_hi = _mm_setzero_ps();
            
            for (int i = 0; i < frames_count; ++i) {
                __m128i src;
                if (sizeof(pixel_t) == 1)
                  src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[i] + x));
                else
                  src = _mm_load_si128(reinterpret_cast<const __m128i*>(src_pointers[i] + x));
                auto weight = _mm_set1_ps(weights[i]);

                if(sizeof(pixel_t) == 1)
                  src = _mm_unpacklo_epi8(src, zero);
                auto src_lo_ps = _mm_cvtepi32_ps(_mm_unpacklo_epi16(src, zero));
                auto src_hi_ps = _mm_cvtepi32_ps(_mm_unpackhi_epi16(src, zero));

                auto weighted_lo = _mm_mul_ps(src_lo_ps, weight);
                auto weighted_hi = _mm_mul_ps(src_hi_ps, weight);
                
                acc_lo = _mm_add_ps(acc_lo, weighted_lo);
                acc_hi = _mm_add_ps(acc_hi, weighted_hi);
            }
            auto dst_lo = _mm_cvtps_epi32(acc_lo);
            auto dst_hi = _mm_cvtps_epi32(acc_hi);

            __m128i dst;
            if (sizeof(pixel_t) == 1) {
              dst = _mm_packs_epi32(dst_lo, dst_hi);
              dst = _mm_packus_epi16(dst, zero);
            }
            else if (sizeof(pixel_t) == 2) {
              if (bits_per_pixel < 16) {
                dst = _mm_packs_epi32(dst_lo, dst_hi); // no need for packus
              }
              else {
                dst = _MM_PACKUS_EPI32(dst_lo, dst_hi); // SSE2 friendly but slower
              }
            }
            
            if (sizeof(pixel_t) == 2 && bits_per_pixel < 16)
              dst = _mm_min_epi16(dst, pixel_limit); // no need for SSE4 epu16 

            if(sizeof(pixel_t) == 1)
              _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp+x), dst);
            else
              _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), dst);
        }

        int start = mod_width / sizeof(pixel_t);
        int end = width / sizeof(pixel_t);
        for (int x = start; x < end; ++x) {
            float acc = 0;
            for (int i = 0; i < frames_count; ++i) {
                acc += reinterpret_cast<const pixel_t *>(src_pointers[i])[x] * weights[i];
            }
            reinterpret_cast<pixel_t *>(dstp)[x] = static_clip<0, max_pixel_value>(acc);
        }

        for (int i = 0; i < frames_count; ++i) {
            src_pointers[i] += src_pitches[i];
        }
        dstp += dst_pitch;
    }
}

static inline void weighted_average_f_sse2(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height) {
  // width is row_size
  int mod_width = width / 16 * 16;

  const int sse_size = 16;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod_width; x += sse_size) {
      __m128 acc = _mm_setzero_ps();

      for (int i = 0; i < frames_count; ++i) {
        __m128 src;
        src = _mm_load_ps(reinterpret_cast<const float*>(src_pointers[i] + x));
        auto weight = _mm_set1_ps(weights[i]);

        auto weighted = _mm_mul_ps(src, weight);

        acc = _mm_add_ps(acc, weighted);
      }

      _mm_store_ps(reinterpret_cast<float*>(dstp + x), acc);
    }

    for (int x = mod_width / 4; x < width / 4; ++x) {
      float acc = 0;
      for (int i = 0; i < frames_count; ++i) {
        acc += reinterpret_cast<const float *>(src_pointers[i])[x] * weights[i];
      }
      reinterpret_cast<float *>(dstp)[x] = acc; // float: no clamping
    }

    for (int i = 0; i < frames_count; ++i) {
      src_pointers[i] += src_pitches[i];
    }
    dstp += dst_pitch;
  }
}


template<int frames_count_2_3_more>
static inline void weighted_average_int_sse2(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height) {
    int16_t *int_weights = reinterpret_cast<int16_t*>(alloca(frames_count*sizeof(int16_t)));
    for (int i = 0; i < frames_count; ++i) {
        int_weights[i] = static_cast<int16_t>((1 << 14) * weights[i]);
    }
    int mod8_width = width / 8 * 8;
    __m128i zero = _mm_setzero_si128();

    __m128i round_mask = _mm_set1_epi32(0x2000);

    bool even_frames = (frames_count % 2 != 0);

    if (frames_count_2_3_more == 2 || frames_count_2_3_more == 3) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < mod8_width; x += 8) {
          __m128i acc_lo = _mm_setzero_si128();
          __m128i acc_hi = _mm_setzero_si128();

          __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[0] + x));
          __m128i src2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[1] + x));
          __m128i weight = _mm_set1_epi32(*reinterpret_cast<int*>(int_weights));

          src = _mm_unpacklo_epi8(src, zero);
          src2 = _mm_unpacklo_epi8(src2, zero);
          __m128i src_lo = _mm_unpacklo_epi16(src, src2);
          __m128i src_hi = _mm_unpackhi_epi16(src, src2);

          __m128i weighted_lo = _mm_madd_epi16(src_lo, weight);
          __m128i weighted_hi = _mm_madd_epi16(src_hi, weight);

          acc_lo = _mm_add_epi32(acc_lo, weighted_lo);
          acc_hi = _mm_add_epi32(acc_hi, weighted_hi);

          if (frames_count_2_3_more == 3) {
            __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[2] + x));
            __m128i weight = _mm_set1_epi32(int_weights[2]);

            src = _mm_unpacklo_epi8(src, zero);
            __m128i src_lo = _mm_unpacklo_epi16(src, zero);
            __m128i src_hi = _mm_unpackhi_epi16(src, zero);

            __m128i weighted_lo = _mm_madd_epi16(src_lo, weight);
            __m128i weighted_hi = _mm_madd_epi16(src_hi, weight);

            acc_lo = _mm_add_epi32(acc_lo, weighted_lo);
            acc_hi = _mm_add_epi32(acc_hi, weighted_hi);
          }

          acc_lo = _mm_add_epi32(acc_lo, round_mask);
          acc_hi = _mm_add_epi32(acc_hi, round_mask);

          __m128i dst_lo = _mm_srai_epi32(acc_lo, 14);
          __m128i dst_hi = _mm_srai_epi32(acc_hi, 14);

          __m128i dst = _mm_packs_epi32(dst_lo, dst_hi);
          dst = _mm_packus_epi16(dst, zero);
          _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x), dst);
        }

        for (int x = mod8_width; x < width; ++x) {
          float acc = 0;
          acc += src_pointers[0][x] * weights[0];
          acc += src_pointers[1][x] * weights[1];
          if (frames_count_2_3_more == 3)
            acc += src_pointers[2][x] * weights[2];
          dstp[x] = static_clip<0, 255>(acc);
        }
       
        src_pointers[0] += src_pitches[0];
        src_pointers[1] += src_pitches[1];
        if (frames_count_2_3_more == 3)
          src_pointers[2] += src_pitches[2];
        dstp += dst_pitch;
      }
    } else {
      // generic path
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < mod8_width; x += 8) {
          __m128i acc_lo = _mm_setzero_si128();
          __m128i acc_hi = _mm_setzero_si128();

          for (int i = 0; i < frames_count - 1; i += 2) {
            __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[i] + x));
            __m128i src2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[i + 1] + x));
            __m128i weight = _mm_set1_epi32(*reinterpret_cast<int*>(int_weights + i));

            src = _mm_unpacklo_epi8(src, zero);
            src2 = _mm_unpacklo_epi8(src2, zero);
            __m128i src_lo = _mm_unpacklo_epi16(src, src2);
            __m128i src_hi = _mm_unpackhi_epi16(src, src2);

            __m128i weighted_lo = _mm_madd_epi16(src_lo, weight);
            __m128i weighted_hi = _mm_madd_epi16(src_hi, weight);

            acc_lo = _mm_add_epi32(acc_lo, weighted_lo);
            acc_hi = _mm_add_epi32(acc_hi, weighted_hi);
          }

          if (even_frames) {
            __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[frames_count - 1] + x));
            __m128i weight = _mm_set1_epi32(int_weights[frames_count - 1]);

            src = _mm_unpacklo_epi8(src, zero);
            __m128i src_lo = _mm_unpacklo_epi16(src, zero);
            __m128i src_hi = _mm_unpackhi_epi16(src, zero);

            __m128i weighted_lo = _mm_madd_epi16(src_lo, weight);
            __m128i weighted_hi = _mm_madd_epi16(src_hi, weight);

            acc_lo = _mm_add_epi32(acc_lo, weighted_lo);
            acc_hi = _mm_add_epi32(acc_hi, weighted_hi);
          }

          acc_lo = _mm_add_epi32(acc_lo, round_mask);
          acc_hi = _mm_add_epi32(acc_hi, round_mask);

          __m128i dst_lo = _mm_srai_epi32(acc_lo, 14);
          __m128i dst_hi = _mm_srai_epi32(acc_hi, 14);

          __m128i dst = _mm_packs_epi32(dst_lo, dst_hi);
          dst = _mm_packus_epi16(dst, zero);
          _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x), dst);
        }

        for (int x = mod8_width; x < width; ++x) {
          float acc = 0;
          for (int i = 0; i < frames_count; ++i) {
            acc += src_pointers[i][x] * weights[i];
          }
          dstp[x] = static_clip<0, 255>(acc);
        }

        for (int i = 0; i < frames_count; ++i) {
          src_pointers[i] += src_pitches[i];
        }
        dstp += dst_pitch;
      }
    }
}
#endif
// INTEL_INTRINSICS end


class AveragePlus : public GenericVideoFilter {
public:
    AveragePlus(std::vector<WeightedClip> clips,
      int ythresh, int uthresh, int vthresh, float scthresh,
      int y, int u, int v, int opt, int pmode, int ythupd, int uthupd, int vthupd,
      int ypnew, int upnew, int vpnew, int threads, int idx_src,
      IScriptEnvironment* env)
      : GenericVideoFilter(clips[0].clip), clips_(clips),
      _thresh{ ythresh, uthresh, vthresh },
      _threshF{ 0.0f, 0.0f, 0.0f },
      _shift(vi.BitsPerComponent() - 8),
      _scthresh(scthresh),
      _opt(opt),
      _pmode(pmode),
      _thUPD{ ythupd, uthupd, vthupd },
      _pnew{ ypnew, upnew, vpnew },
      _threads{ threads },
      _idx_th_src(idx_src)
    {

      has_at_least_v8 = true;
      try { env->CheckVersion(8); }
      catch (const AvisynthError&) { has_at_least_v8 = false; }

      if (ythresh < 1 || ythresh > 256)
        env->ThrowError("AveragePlus: ythresh must be between 1..256.");
      if (uthresh < 1 || uthresh > 256)
        env->ThrowError("AveragePlus: uthresh must be between 1..256.");
      if (vthresh < 1 || vthresh > 256)
        env->ThrowError("AveragePlus: vthresh must be between 1..256.");
      if (_scthresh < 0.f || _scthresh > 100.f)
        env->ThrowError("AveragePlus: scthresh must be between 0.0..100.0.");
      if (_opt < -1 || _opt > 3)
        env->ThrowError("AveragePlus: opt must be between -1..3.");
      if (ythupd < 0)
        env->ThrowError("AveragePlus: ythupd must be greater than 0.");
      if (uthupd < 0)
        env->ThrowError("AveragePlus: uthupd must be greater than 0.");
      if (vthupd < 0)
        env->ThrowError("AveragePlus: vthupd must be greater than 0.");
      if (ypnew < 0)
        env->ThrowError("AveragePlus: ypnew must be greater than 0.");
      if (upnew < 0)
        env->ThrowError("AveragePlus: upnew must be greater than 0.");
      if (vpnew < 0)
        env->ThrowError("AveragePlus: vpnew must be greater than 0.");

      int frames_count = (int)clips_.size();
      _num_clips = frames_count; // save to global ?

      int pixelsize = vi.ComponentSize();
      int bits_per_pixel = vi.BitsPerComponent();

  #ifdef INTEL_INTRINSICS
      const bool avx = !!(env->GetCPUFlags() & CPUF_AVX);
      const bool avx2 = !!(env->GetCPUFlags() & CPUF_AVX2);
  #endif
      // we don't know the alignment here. avisynth+: 32 bytes, classic: 16
      // decide later (processor_, processor_32aligned)

  #ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_SSE2) {
        bool use_weighted_average_f = false;
        if (pixelsize == 1) {
          if (frames_count == 2)
            processor_ = &weighted_average_int_sse2<2>;
          else if (frames_count == 3)
            processor_ = &weighted_average_int_sse2<3>;
          else
            processor_ = &weighted_average_int_sse2<0>;
          processor_32aligned_ = processor_;
          for (const auto& clip : clips) {
            if (std::abs(clip.weight) > 1) {
              use_weighted_average_f = true;
              break;
            }
          }
          if (clips.size() > 255) {
            // too many clips, may overflow
            use_weighted_average_f = true;
          }
        }
        else {
          // uint16 and float: float mode internally
          use_weighted_average_f = true;
        }

        if (use_weighted_average_f) {
          switch (bits_per_pixel) {
          case 8:
            processor_ = &weighted_average_sse2<uint8_t, 8>;
            processor_32aligned_ = avx2 ? &weighted_average_avx2<uint8_t, 8> : avx ? &weighted_average_avx<uint8_t, 8> : &weighted_average_sse2<uint8_t, 8>;
            break;
          case 10:
            processor_ = &weighted_average_sse2<uint16_t, 10>;
            processor_32aligned_ = avx2 ? &weighted_average_avx2<uint16_t, 10> : avx ? &weighted_average_avx<uint16_t, 10> : &weighted_average_sse2<uint16_t, 10>;
            break;
          case 12:
            processor_ = &weighted_average_sse2<uint16_t, 12>;
            processor_32aligned_ = avx2 ? &weighted_average_avx2<uint16_t, 12> : avx ? &weighted_average_avx<uint16_t, 12> : &weighted_average_sse2<uint16_t, 12>;
            break;
          case 14:
            processor_ = &weighted_average_sse2<uint16_t, 14>;
            processor_32aligned_ = avx2 ? &weighted_average_avx2<uint16_t, 14> : avx ? &weighted_average_avx<uint16_t, 14> : &weighted_average_sse2<uint16_t, 14>;
            break;
          case 16:
            processor_ = &weighted_average_sse2<uint16_t, 16>;
            processor_32aligned_ = avx2 ? &weighted_average_avx2<uint16_t, 16> : avx ? &weighted_average_avx<uint16_t, 16> : processor_;
            break;
          case 32:
            processor_ = &weighted_average_f_sse2;
            processor_32aligned_ = avx2 ? &weighted_average_f_avx2 : avx ? &weighted_average_f_avx : &weighted_average_f_sse2;
            break;
          }
        }
      }
      else
  #endif
      {
        switch (bits_per_pixel) {
        case 8:
          processor_ = &weighted_average_c<uint8_t, 8>;
          break;
        case 10:
          processor_ = &weighted_average_c<uint16_t, 10>;
          break;
        case 12:
          processor_ = &weighted_average_c<uint16_t, 12>;
          break;
        case 14:
          processor_ = &weighted_average_c<uint16_t, 14>;
          break;
        case 16:
          processor_ = &weighted_average_c<uint16_t, 16>;
          break;
        case 32:
          processor_ = &weighted_average_c<float, 1>; // bits_per_pixel n/a
          break;
        }
        processor_32aligned_ = processor_;
      }

      const int planes[3]{ y, u, v };
      static constexpr int iMaxSum{ std::numeric_limits<int>::max() };
      static constexpr float fMaxSum{ std::numeric_limits<float>::max() };

      for (int i{ 0 }; i < std::min(vi.NumComponents(), 3); ++i)
      {
        switch (planes[i])
        {
        case 3:
          proccesplanes[i] = 3;
          break;
        case 2:
          proccesplanes[i] = 2;
          break;
        case 1:
          proccesplanes[i] = 1;
          break;
        default:
          env->ThrowError("AveragePlus: y / u / v must be between 1..3.");
        }

        if (proccesplanes[i] == 3)
        {

          if (_pmode == 1 && _thUPD[i] > 0)
          {
            const size_t num_elements_minsum{ static_cast<size_t>(vi.width) * vi.height };
            pIIRMem[i].resize(num_elements_minsum * vi.ComponentSize(), 0);
            pMinSumMem[i].resize(num_elements_minsum, (vi.ComponentSize() < 4) ? iMaxSum : fMaxSum);
          }
        }
      }

    }
    
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  template<typename T>
  void filter_mode2_C(PVideoFrame src[MAX_CLIPS], PVideoFrame& dst, const int plane);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  std::vector<WeightedClip> clips_;
  decltype(&weighted_average_c<uint8_t, 8>) processor_;
  decltype(&weighted_average_c<uint8_t, 8>) processor_32aligned_;
  bool has_at_least_v8;

  int _num_clips;
  int _idx_th_src;
  int _thresh[3];
  float _threshF[3];
  float _scthresh;
  int _shift;
  int proccesplanes[3];
  int _opt;

  int _pmode;
  std::array<std::vector<uint8_t>, 3> pIIRMem;
  int _thUPD[3];
  std::array<std::vector<std::variant<int, float>>, 3> pMinSumMem;
  int _pnew[3];
  int _threads;

#ifdef _DEBUG
  // MEL debug stat
  int iMEL_mem_hits;
  int iMEL_mem_updates;
#endif

};

template<typename T>
void AveragePlus::filter_mode2_C(PVideoFrame src[MAX_CLIPS], PVideoFrame& dst, const int plane)
{
  int src_stride[MAX_CLIPS]{};
  int pf_stride[MAX_CLIPS]{};
  const size_t stride{ dst->GetPitch(plane) / sizeof(T) };
  const int width{ static_cast<int>(dst->GetRowSize(plane) / sizeof(T)) };
  const int height{ dst->GetHeight(plane) };
  const T* g_srcp[MAX_CLIPS]{};

  const int l{ plane >> 1 };

  typedef typename std::conditional<sizeof(T) <= 2, int, float>::type working_t;

  const working_t thresh = (sizeof(T) <= 2) ? (_thresh[l] << _shift) : (_thresh[l] / 256.0f);

  const working_t thUPD = (sizeof(T) <= 2) ? (_thUPD[l] << _shift) : (_thUPD[l] / 256.0f);
  const working_t pnew = (sizeof(T) <= 2) ? (_pnew[l] << _shift) : (_pnew[l] / 256.0f);
  T* g_pMem{ reinterpret_cast<T*>(pIIRMem[l].data()) };
  working_t* g_pMemSum{ reinterpret_cast<working_t*>(pMinSumMem[l].data()) };
  const working_t MaxSumDM = (sizeof(T) < 2) ? 255 * (_num_clips) : 65535 * (_num_clips); // 65535 is enough max for float too

  for (int i{ 0 }; i < _num_clips; ++i)
  {
    src_stride[i] = src[i]->GetPitch(plane) / sizeof(T);
    g_srcp[i] = reinterpret_cast<const T*>(src[i]->GetReadPtr(plane));
  }

  T* g_dstp{ reinterpret_cast<T*>(dst->GetWritePtr(plane)) };

#ifdef _DEBUG
  iMEL_mem_hits = 0;
#endif

#pragma omp parallel for num_threads(_threads)
  for (int y = 0; y < height; ++y)
  {
    // local threads ptrs
    const T* srcp[MAX_CLIPS]{};
    T* dstp, * pMem;
    working_t* pMemSum;

    for (int i{ 0 }; i < _num_clips; ++i)
    {
      srcp[i] = g_srcp[i] + y * src_stride[i];
    }

    dstp = g_dstp + y * stride;
    pMem = g_pMem + y * width;
    pMemSum = g_pMemSum + y * width;

    for (int x{ 0 }; x < width; ++x)
    {
      // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
      working_t wt_sum_minrow = MaxSumDM;
      int i_idx_minrow = 0;

      for (int dmt_row = 0; dmt_row < (_num_clips); dmt_row++)
      {
        working_t wt_sum_row = 0;
        for (int dmt_col = 0; dmt_col < (_num_clips); dmt_col++)
        {
          if (dmt_row == dmt_col)
          { // block with itself => DM=0
            continue;
          }

          T* row_data_ptr;
          T* col_data_ptr;

          row_data_ptr = (T*)&srcp[dmt_row][x];
          col_data_ptr = (T*)&srcp[dmt_col][x];

          wt_sum_row += (sizeof(T) <= 2) ? std::abs(*row_data_ptr - *col_data_ptr) : std::abs(*row_data_ptr - *col_data_ptr); // why equal ?
        }

        if (wt_sum_row < wt_sum_minrow)
        {
          wt_sum_minrow = wt_sum_row;
          i_idx_minrow = dmt_row;
        }
      }

      // set block of idx_minrow as output block
      const T* best_data_ptr;

      best_data_ptr = &srcp[i_idx_minrow][x];

      if (thUPD > 0) // IIR here
      {
        // IIR - check if memory sample is still good
        working_t idm_mem = (sizeof(T) <= 2) ? std::abs(*best_data_ptr - pMem[x]) : std::abs(*best_data_ptr - pMem[x]);

        if ((idm_mem < thUPD) && ((wt_sum_minrow + pnew) > pMemSum[x]))
        {
          // mem still good - output mem block
          best_data_ptr = &pMem[x];

#ifdef _DEBUG
          iMEL_mem_hits++;
#endif
        }
        else // mem no good - update mem
        {
          pMem[x] = *best_data_ptr;
          pMemSum[x] = wt_sum_minrow;
        }
      }

      // if any input clip marked as thresh-source, use thresh, if (_idx_src == -1) - no thresh-source marked
      if (_idx_th_src > -1)
      {
        // check if best is below thresh-difference from current src
        if (((sizeof(T) <= 2) ? std::abs(*best_data_ptr - srcp[_idx_th_src][x]) : std::abs(*best_data_ptr - srcp[_idx_th_src][x])) < thresh)
        {
          dstp[x] = *best_data_ptr;
        }
        else
        {
          dstp[x] = srcp[_idx_th_src][x];
        }
      }
      else
        dstp[x] = *best_data_ptr;
    }
  }

#ifdef _DEBUG
  float fRatioMEL_mem_samples = (float)iMEL_mem_hits / (float)(width * height);
  int idbr = 0;
#endif
}


PVideoFrame AveragePlus::GetFrame(int n, IScriptEnvironment* env) {
  int frames_count = (int)clips_.size();
  PVideoFrame* src_frames = reinterpret_cast<PVideoFrame*>(alloca(frames_count * sizeof(PVideoFrame)));
  const uint8_t** src_ptrs = reinterpret_cast<const uint8_t**>(alloca(sizeof(uint8_t*) * frames_count));
  int* src_pitches = reinterpret_cast<int*>(alloca(sizeof(int) * frames_count));
  float* weights = reinterpret_cast<float*>(alloca(sizeof(float) * frames_count));
  if (src_pitches == nullptr || src_frames == nullptr || src_ptrs == nullptr || weights == nullptr) {
    env->ThrowError("Average: Couldn't allocate memory on stack. This is a bug, please report");
  }

  PVideoFrame src[MAX_CLIPS]{};
  PVideoFrame dst;

  if (_pmode == 1)
  {
    for (int i{ 0 }; i < frames_count; ++i)
    {
      src[i] = clips_[i].clip->GetFrame(n, env);
    }

    // frame props from the first clip
    dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &src[0]) : env->NewVideoFrame(vi);

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int* planes = (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) ? planes_r : planes_y;

    for (int i{ 0 }; i < std::min(vi.NumComponents(), 3); ++i)
    {
      if (proccesplanes[i] == 3)
      {

        switch (vi.ComponentSize())
        {
          case 1: {
            filter_mode2_C<uint8_t>(src, dst, planes[i]);
            break;
          }
          case 2: {
            filter_mode2_C<uint16_t>(src, dst, planes[i]);
            break;
          }
          default: {
            filter_mode2_C<float>(src, dst, planes[i]);
          }
          continue;
        }
      }
    }
    return dst; // end of pmode=1
  }

  if (_pmode == 0)
  {

    memset(src_frames, 0, frames_count * sizeof(PVideoFrame));

    for (int i = 0; i < frames_count; ++i) {
      src_frames[i] = clips_[i].clip->GetFrame(n, env);
      weights[i] = clips_[i].weight;
    }

    // frame props from the first clip
    /*PVideoFrame*/ dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &src_frames[0]) : env->NewVideoFrame(vi);

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int* planes = (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) ? planes_r : planes_y;

    bool hasAlpha = vi.IsPlanarRGBA() || vi.IsYUVA();

    for (int pid = 0; pid < (vi.IsY() ? 1 : (hasAlpha ? 4 : 3)); pid++) {
      int plane = planes[pid];
      int width = dst->GetRowSize(plane);
      int height = dst->GetHeight(plane);
      auto dstp = dst->GetWritePtr(plane);
      int dst_pitch = dst->GetPitch(plane);

      bool allSrc32aligned = true;
      for (int i = 0; i < frames_count; ++i) {
        src_ptrs[i] = src_frames[i]->GetReadPtr(plane);
        src_pitches[i] = src_frames[i]->GetPitch(plane);
        if (!IsPtrAligned(src_ptrs[i], 32))
          allSrc32aligned = false;
        if (src_pitches[i] & 0x1F)
          allSrc32aligned = false;
      }

      if (IsPtrAligned(dstp, 32) && (dst_pitch & 0x1F) == 0 && allSrc32aligned)
        processor_32aligned_(dstp, dst_pitch, src_ptrs, src_pitches, weights, frames_count, width, height);
      else
        processor_(dstp, dst_pitch, src_ptrs, src_pitches, weights, frames_count, width, height);
    }

    for (int i = 0; i < frames_count; ++i) {
      src_frames[i].~PVideoFrame();
    }
  }

    return dst;
}


AVSValue __cdecl create_average_plus(AVSValue args, void* user_data, IScriptEnvironment* env) {

  enum
  {
    Clip, // as list of clips and weights ?
    Ythresh,
    Uthresh,
    Vthresh,
    Scthresh,
    Y,
    U,
    V,
    Opt,
    Pmode,
    YthUPD,
    UthUPD,
    VthUPD,
    Ypnew,
    Upnew,
    Vpnew,
    Threads,
    idx_src // optional index of the 'source' clip, default to -1 - no special source selected (*thresh not applicable)
  };

    AVSValue args0 = args[0];
    int arguments_count = args0.ArraySize();
    if (arguments_count == 1 && args0[0].IsArray()) {
      args0 = args0[0];
      arguments_count = args0.ArraySize();
    }

    if (arguments_count % 2 != 0) {
        env->ThrowError("Average requires an even number of arguments (clip,weight,...) listed or passed in a single array.");
    }
    if (arguments_count == 0) {
        env->ThrowError("Average: At least one clip has to be supplied.");
    }
    std::vector<WeightedClip> clips;
    auto first_clip = args0[0].AsClip();
    auto first_vi = first_clip->GetVideoInfo();
    clips.emplace_back(first_clip, static_cast<float>(args0[1].AsFloat()));

    for (int i = 2; i < arguments_count; i += 2) {
        auto clip = args0[i].AsClip();
        float weight = static_cast<float>(args0[i+1].AsFloat());
        if (std::abs(weight) < 0.00001f) {
            continue;
        }
        auto vi = clip->GetVideoInfo();
        if (!vi.IsSameColorspace(first_vi)) {
            env->ThrowError("Average: all clips must have the same colorspace.");
        }
        if (vi.width != first_vi.width || vi.height != first_vi.height) {
            env->ThrowError("Average: all clips must have identical width and height.");
        }
        if (vi.num_frames < first_vi.num_frames) {
            env->ThrowError("Average: all clips must be have same or greater number of frames as the first one.");
        }

        clips.emplace_back(clip, weight);
    }

  //    return new AveragePlus(clips, env);
  return new AveragePlus(clips,
    args[Ythresh].AsInt(4), args[Uthresh].AsInt(5), args[Vthresh].AsInt(5),
    args[Scthresh].AsFloatf(12), args[Y].AsInt(3), args[U].AsInt(3), args[V].AsInt(3), args[Opt].AsInt(-1),
    args[Pmode].AsInt(0), args[YthUPD].AsInt(0), args[UthUPD].AsInt(0), args[VthUPD].AsInt(0), args[Ypnew].AsInt(0),
    args[Upnew].AsInt(0), args[Vpnew].AsInt(0), args[Threads].AsInt(1), args[idx_src].AsInt(-1),    
    env);
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    env->AddFunction("AveragePlus", ".*[ythresh]i[uthresh]i[vthresh]i[scthresh]f[y]i[u]i[v]i[opt]i[pmode]"
        "i[ythupd]i[uthupd]i[vthupd]i[ypnew]i[upnew]i[vpnew]i[threads]i[idx_src]i", create_average_plus, 0);
    return "Mind your sugar level";
}
