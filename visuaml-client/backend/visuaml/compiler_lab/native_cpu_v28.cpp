// VisuaML native CPU research dialect: conventional bounded Llama forward.
// This implementation uses only ISO C++17 and libm. No PyTorch, NumPy,
// TorchScript, BLAS, or Python mathematical execution enters this code.
// Python must validate the exact checkpoint, tensor shapes and capacity
// before calling. All caller-provided pointers are untrusted at the ABI
// boundary; this kernel is NOT a sandbox for hostile native callers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(VISUAML_NATIVE_AVX2_RUNTIME) && \
    (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define VISUAML_HAS_OPTIONAL_AVX2 1
#else
#define VISUAML_HAS_OPTIONAL_AVX2 0
#endif

extern "C" {

struct VisuaMLLlamaConfig {
  int batch;
  int layers;
  int hidden;
  int heads;
  int kv_heads;
  int head_dim;
  int inner;
  int vocabulary;
  int capacity;
  float epsilon;
  double rope_theta;
  int tied_head;
};

int visuaml_native_avx2_active() noexcept;

int visuaml_native_llama_decode(
    const VisuaMLLlamaConfig* c,
    const int64_t* tokens,
    int query,
    int past,
    const float* const* weights,
    const float* const* previous_kv,
    float* const* emitted_kv,
    float* logits) noexcept;

} // extern "C"

namespace {

// AVX2 lives in a target-specific function, not a global -mavx2 binary.
// The runtime guard ensures that binaries remain runnable on non-AVX2 CPUs.
inline bool native_avx2_available() noexcept {
#if VISUAML_HAS_OPTIONAL_AVX2
  static const bool supported = __builtin_cpu_supports("avx2") != 0;
  return supported;
#else
  return false;
#endif
}

#if VISUAML_HAS_OPTIONAL_AVX2
__attribute__((target("avx2")))
void linear_avx2(const float* input, const float* weight, float* output,
                 int input_width, int output_width) {
  int row = 0;
  for (; row + 3 < output_width; row += 4) {
    const float* w0 = weight + static_cast<size_t>(row) * input_width;
    const float* w1 = w0 + input_width;
    const float* w2 = w1 + input_width;
    const float* w3 = w2 + input_width;
    __m256 a = _mm256_setzero_ps();
    __m256 b = _mm256_setzero_ps();
    __m256 c = _mm256_setzero_ps();
    __m256 d = _mm256_setzero_ps();
    int col = 0;
    for (; col + 7 < input_width; col += 8) {
      const __m256 x = _mm256_loadu_ps(input + col);
      a = _mm256_add_ps(a, _mm256_mul_ps(x, _mm256_loadu_ps(w0 + col)));
      b = _mm256_add_ps(b, _mm256_mul_ps(x, _mm256_loadu_ps(w1 + col)));
      c = _mm256_add_ps(c, _mm256_mul_ps(x, _mm256_loadu_ps(w2 + col)));
      d = _mm256_add_ps(d, _mm256_mul_ps(x, _mm256_loadu_ps(w3 + col)));
    }
    alignas(32) float lane[8];
    auto reduce = [&lane](__m256 value) {
      _mm256_store_ps(lane, value);
      float sum = 0.0f;
      for (int i = 0; i < 8; ++i) sum += lane[i];
      return sum;
    };
    float sums[4] = {reduce(a), reduce(b), reduce(c), reduce(d)};
    for (; col < input_width; ++col) {
      const float x = input[col];
      sums[0] += w0[col] * x;
      sums[1] += w1[col] * x;
      sums[2] += w2[col] * x;
      sums[3] += w3[col] * x;
    }
    for (int i = 0; i < 4; ++i) output[row + i] = sums[i];
  }
  for (; row < output_width; ++row) {
    const float* w = weight + static_cast<size_t>(row) * input_width;
    __m256 accumulator = _mm256_setzero_ps();
    int col = 0;
    for (; col + 7 < input_width; col += 8) {
      accumulator = _mm256_add_ps(
          accumulator,
          _mm256_mul_ps(_mm256_loadu_ps(w + col),
                        _mm256_loadu_ps(input + col)));
    }
    alignas(32) float lane[8];
    _mm256_store_ps(lane, accumulator);
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) sum += lane[i];
    for (; col < input_width; ++col) sum += w[col] * input[col];
    output[row] = sum;
  }
}
#endif

// Preserve a separately compilable scalar reference for differential testing.
// The default path processes four output rows per input traversal. The per-row
// accumulation order stays identical; no fast-math reassociation is enabled.
inline void linear(const float* input, const float* weight,
                   float* output, int input_width, int output_width) {
#if VISUAML_HAS_OPTIONAL_AVX2
  // Short widths favor scalar traversal; dispatch cost dominates there.
  if (input_width >= 32 && native_avx2_available()) {
    linear_avx2(input, weight, output, input_width, output_width);
    return;
  }
#endif
#if defined(VISUAML_NATIVE_SCALAR_REFERENCE)
  for (int row = 0; row < output_width; ++row) {
    float sum = 0.0f;
    const float* w = weight + static_cast<size_t>(row) * input_width;
    for (int col = 0; col < input_width; ++col) {
      sum += w[col] * input[col];
    }
    output[row] = sum;
  }
#else
  int row = 0;
  for (; row + 3 < output_width; row += 4) {
    const float* w0 = weight + static_cast<size_t>(row) * input_width;
    const float* w1 = w0 + input_width;
    const float* w2 = w1 + input_width;
    const float* w3 = w2 + input_width;
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    for (int col = 0; col < input_width; ++col) {
      const float x = input[col];
      a += w0[col] * x;
      b += w1[col] * x;
      c += w2[col] * x;
      d += w3[col] * x;
    }
    output[row] = a;
    output[row + 1] = b;
    output[row + 2] = c;
    output[row + 3] = d;
  }
  for (; row < output_width; ++row) {
    float sum = 0.0f;
    const float* w = weight + static_cast<size_t>(row) * input_width;
    for (int col = 0; col < input_width; ++col) {
      sum += w[col] * input[col];
    }
    output[row] = sum;
  }
#endif
}

inline void rmsnorm(const float* input, const float* gain, float* output,
                    int width, float epsilon) {
  float sum = 0.0f;
  for (int i = 0; i < width; ++i) sum += input[i] * input[i];
  const float factor = 1.0f / std::sqrt(sum / width + epsilon);
  for (int i = 0; i < width; ++i) output[i] = input[i] * factor * gain[i];
}

// Cache the RoPE angles once per position, shared across Q/K heads.
inline void rope_angles(int position, int depth, double theta,
                        std::vector<float>& sine, std::vector<float>& cosine) {
  for (int i = 0; i < depth / 2; ++i) {
    const double phase =
        position * std::pow(theta, -2.0 * static_cast<double>(i) / depth);
    sine[i] = static_cast<float>(std::sin(phase));
    cosine[i] = static_cast<float>(std::cos(phase));
  }
}

inline void rotate_half(float* value, int depth,
                        const std::vector<float>& sine,
                        const std::vector<float>& cosine) {
  const int half = depth / 2;
  for (int i = 0; i < half; ++i) {
    const float a = value[i];
    const float b = value[i + half];
    value[i] = a * cosine[i] - b * sine[i];
    value[i + half] = b * cosine[i] + a * sine[i];
  }
}

inline size_t kv_index(int b, int t, int h, int d,
                       int context, int kv_heads, int depth) {
  return (((static_cast<size_t>(b) * context + t) * kv_heads + h) *
          depth + d);
}

inline bool admitted(const VisuaMLLlamaConfig& c, int query, int past) {
  return c.batch >= 1 && c.batch <= 2 &&
         c.layers >= 1 && c.layers <= 8 &&
         c.hidden >= 8 && c.hidden <= 384 &&
         c.heads >= 1 && c.heads <= 16 &&
         c.kv_heads >= 1 && c.kv_heads <= 16 &&
         c.hidden % c.heads == 0 && c.heads % c.kv_heads == 0 &&
         c.head_dim == c.hidden / c.heads && c.head_dim % 2 == 0 &&
         c.inner >= 8 && c.inner <= 2048 &&
         c.vocabulary >= 2 && c.vocabulary <= 32768 &&
         c.capacity >= 1 && c.capacity <= 128 &&
         c.epsilon > 0.0f && c.epsilon <= 0.001f &&
         std::isfinite(c.rope_theta) &&
         c.rope_theta >= 1.0 && c.rope_theta <= 1.0e7 &&
         (c.tied_head == 0 || c.tied_head == 1) &&
         query >= 1 && query <= 32 && past >= 0 &&
         past + query <= c.capacity;
}

}  // namespace

extern "C" int visuaml_native_avx2_active() noexcept {
  return native_avx2_available() ? 1 : 0;
}

extern "C" int visuaml_native_llama_decode(
    const VisuaMLLlamaConfig* config, const int64_t* tokens,
    int query, int past, const float* const* weights,
    const float* const* previous_kv, float* const* emitted_kv,
    float* logits) noexcept {
  try {
    if (!config || !tokens || !weights || !previous_kv ||
        !emitted_kv || !logits) return -1;
    const auto& c = *config;
    if (!admitted(c, query, past)) return -2;
    const int base = c.tied_head ? 2 : 3;
    for (int i = 0; i < base + 9 * c.layers; ++i) {
      if (!weights[i]) return -3;
    }
    for (int i = 0; i < 2 * c.layers; ++i) {
      if (!previous_kv[i] || !emitted_kv[i]) return -4;
    }
    const size_t rows = static_cast<size_t>(c.batch) * query;
    for (size_t row = 0; row < rows; ++row) {
      if (tokens[row] < 0 || tokens[row] >= c.vocabulary) return -5;
    }
    const int h = c.hidden;
    const int depth = c.head_dim;
    const int kwidth = c.kv_heads * depth;
    const float scaling = 1.0f / std::sqrt(static_cast<float>(depth));
    const int heads_per_kv = c.heads / c.kv_heads;
    std::vector<float> state(rows * h);
    // Reuse scratch across layers and heads, eliminating repeated allocations.
    const size_t kv_len = rows * kwidth;
    std::vector<float> normed(rows * h);
    std::vector<float> q(rows * h);
    std::vector<float> attended(rows * h);
    std::vector<float> keys(kv_len);
    std::vector<float> values(kv_len);
    std::vector<float> rope_sine(depth / 2);
    std::vector<float> rope_cosine(depth / 2);
    std::vector<float> scores(c.capacity);
    std::vector<float> projection(h);
    std::vector<float> ff_norm(h);
    std::vector<float> gate(c.inner);
    std::vector<float> up(c.inner);
    std::vector<float> product(c.inner);
    std::vector<float> ff_out(h);
    std::vector<float> normalized(h);

    for (size_t row = 0; row < rows; ++row) {
      const float* start = weights[0] + tokens[row] * h;
      std::copy(start, start + h, state.data() + row * h);
    }

    for (int layer = 0; layer < c.layers; ++layer) {
      const int index = base + layer * 9;
      const float* norm_attn = weights[index + 0];
      const float* wq = weights[index + 1];
      const float* wk = weights[index + 2];
      const float* wv = weights[index + 3];
      const float* wo = weights[index + 4];
      const float* norm_ff = weights[index + 5];
      const float* wgate = weights[index + 6];
      const float* wup = weights[index + 7];
      const float* wdown = weights[index + 8];

      std::fill(attended.begin(), attended.end(), 0.0f);
      for (size_t row = 0; row < rows; ++row) {
        const size_t off = row * h;
        const size_t kv_off = row * kwidth;
        rmsnorm(state.data() + off, norm_attn, normed.data() + off,
                h, c.epsilon);
        linear(normed.data() + off, wq, q.data() + off, h, h);
        linear(normed.data() + off, wk, keys.data() + kv_off, h, kwidth);
        linear(normed.data() + off, wv, values.data() + kv_off, h, kwidth);
        const int position = past + static_cast<int>(row % query);
        rope_angles(position, depth, c.rope_theta, rope_sine, rope_cosine);
        for (int head = 0; head < c.heads; ++head) {
          rotate_half(q.data() + off + head * depth,
                      depth, rope_sine, rope_cosine);
        }
        for (int head = 0; head < c.kv_heads; ++head) {
          rotate_half(keys.data() + kv_off + head * depth,
                      depth, rope_sine, rope_cosine);
        }
      }
      std::copy(keys.begin(), keys.end(), emitted_kv[2 * layer]);
      std::copy(values.begin(), values.end(), emitted_kv[2 * layer + 1]);

      const float* old_keys = previous_kv[2 * layer];
      const float* old_values = previous_kv[2 * layer + 1];
      for (int batch = 0; batch < c.batch; ++batch) {
        for (int t = 0; t < query; ++t) {
          const int available = past + t + 1;
          for (int head = 0; head < c.heads; ++head) {
            const int kh = head / heads_per_kv;
            const float* query_head =
                q.data() + (static_cast<size_t>(batch) * query + t) * h
                + head * depth;
            float peak = -std::numeric_limits<float>::infinity();
            for (int ctx = 0; ctx < available; ++ctx) {
              const float* key_ptr =
                  ctx < past
                  ? old_keys + kv_index(batch, ctx, kh, 0, past, c.kv_heads, depth)
                  : keys.data() +
                    kv_index(batch, ctx - past, kh, 0, query, c.kv_heads, depth);
              float dot = 0.0f;
              for (int d = 0; d < depth; ++d) {
                dot += query_head[d] * key_ptr[d];
              }
              scores[ctx] = dot * scaling;
              peak = std::max(peak, scores[ctx]);
            }
            float denominator = 0.0f;
            for (int ctx = 0; ctx < available; ++ctx) {
              scores[ctx] = std::exp(scores[ctx] - peak);
              denominator += scores[ctx];
            }
            if (!(denominator > 0.0f) || !std::isfinite(denominator)) return -6;
            float* output_head =
                attended.data() + (static_cast<size_t>(batch) * query + t) * h
                + head * depth;
            for (int ctx = 0; ctx < available; ++ctx) {
              const float* value_ptr =
                  ctx < past
                  ? old_values + kv_index(batch, ctx, kh, 0, past, c.kv_heads, depth)
                  : values.data() +
                    kv_index(batch, ctx - past, kh, 0, query, c.kv_heads, depth);
              const float probability = scores[ctx] / denominator;
              for (int d = 0; d < depth; ++d) {
                output_head[d] += probability * value_ptr[d];
              }
            }
          }
        }
      }

      for (size_t row = 0; row < rows; ++row) {
        const size_t off = row * h;
        linear(attended.data() + off, wo, projection.data(), h, h);
        for (int d = 0; d < h; ++d) state[off + d] += projection[d];
        rmsnorm(state.data() + off, norm_ff, ff_norm.data(), h, c.epsilon);
        linear(ff_norm.data(), wgate, gate.data(), h, c.inner);
        linear(ff_norm.data(), wup, up.data(), h, c.inner);
        for (int i = 0; i < c.inner; ++i) {
          product[i] = gate[i] * (1.0f / (1.0f + std::exp(-gate[i]))) * up[i];
        }
        linear(product.data(), wdown, ff_out.data(), c.inner, h);
        for (int d = 0; d < h; ++d) state[off + d] += ff_out[d];
      }
    }

    const float* final_gain = weights[1];
    const float* lm_head = weights[c.tied_head ? 0 : 2];
    for (size_t row = 0; row < rows; ++row) {
      rmsnorm(state.data() + row * h, final_gain, normalized.data(), h, c.epsilon);
      linear(normalized.data(), lm_head, logits + row * c.vocabulary,
             h, c.vocabulary);
    }
    for (size_t i = 0; i < rows * c.vocabulary; ++i) {
      if (!std::isfinite(logits[i])) return -7;
    }
    return 0;
  } catch (...) {
    return -99;
  }
}
