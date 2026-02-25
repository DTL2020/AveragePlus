#pragma once

#include "avisynth.h"
#include "avs/alignment.h"
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <array>
#include <variant>

#define MAX_CLIPS 128

#define SIMD_AVX512_SPP 64
#define SIMD_AVX2_SPP 32

struct WeightedClip {
  PClip clip;
  float weight;

  WeightedClip(PClip _clip, float _weight) : clip(_clip), weight(_weight) {}
};

