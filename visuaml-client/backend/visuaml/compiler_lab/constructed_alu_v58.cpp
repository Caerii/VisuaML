#include <cstdint>
#include <cmath>
#include <cstddef>

namespace {
constexpr int width = 32;

double relu(double value) {
    return value < 0.0 ? 0.0 : value;
}

double and_gate(double a, double b, const double* weight) {
    return relu(weight[0] * a + weight[1] * b + weight[2]);
}

double xor_gate(double a, double b, const double* weight) {
    return relu(weight[0] * a + weight[1] * b)
         + relu(weight[2] * a + weight[3] * b);
}

double or_gate(double a, double b, const double* weight) {
    return a + b - and_gate(a, b, weight);
}

void add(const double* a, const double* b, const double* thresholds,
         bool subtract, double* output) {
    double carry = subtract ? 1.0 : 0.0;
    for (int i = 0; i < width; ++i) {
        const double digit = subtract ? 1.0 - b[i] : b[i];
        const double n = a[i] + digit + carry;
        carry = relu(n - thresholds[0]) - relu(n - thresholds[1]);
        output[i] = n - 2.0 * carry;
    }
}

void multiply(const double* a, const double* b, const double* thresholds,
              std::size_t count, double* output) {
    double carry = 0.0;
    for (int column = 0; column < width; ++column) {
        double partial = 0.0;
        for (int i = 0; i <= column; ++i) {
            partial += relu(a[i]) * b[column - i];
        }
        const double n = partial + carry;
        carry = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            carry += relu(n - (thresholds[i] - 1.0)) - relu(n - thresholds[i]);
        }
        output[column] = n - 2.0 * carry;
    }
}

double signed_lt(const double* a, const double* b,
                 const double* and_weight, const double* xor_weight) {
    double same = 1.0;
    double less = 0.0;
    for (int i = width - 1; i >= 0; --i) {
        const double bit_lt = and_gate(1.0 - a[i], b[i], and_weight);
        less = or_gate(less, and_gate(same, bit_lt, and_weight), and_weight);
        same = and_gate(same, 1.0 - xor_gate(a[i], b[i], xor_weight), and_weight);
    }
    const double a_sign = a[width - 1];
    const double b_sign = b[width - 1];
    const double different = xor_gate(a_sign, b_sign, xor_weight);
    return or_gate(and_gate(a_sign, 1.0 - b_sign, and_weight),
                   and_gate(1.0 - different, less, and_weight), and_weight);
}
} // namespace

extern "C" void visuaml_alu_candidates(
        const double* a, const double* b, const double* and_weight,
        const double* xor_weight, const double* carry_thresholds,
        std::size_t carry_count, const double* adder_thresholds,
        double* candidates) {
    add(a, b, adder_thresholds, false, candidates);
    add(a, b, adder_thresholds, true, candidates + width);
    multiply(a, b, carry_thresholds, carry_count, candidates + 2 * width);
    candidates[3 * width] = signed_lt(a, b, and_weight, xor_weight);
    candidates[4 * width] = signed_lt(b, a, and_weight, xor_weight);
    double eqz = 1.0;
    for (int i = 0; i < width; ++i) {
        eqz = and_gate(eqz, 1.0 - a[i], and_weight);
    }
    candidates[5 * width] = eqz;
}

extern "C" int visuaml_compact_route(
        const double* pc, std::size_t slots, const double* condition,
        const std::int32_t* targets, const double* guards,
        const double* guard_weights, const double* path_weights,
        const double* and_weight, double* details) {
    std::size_t active = slots;
    for (std::size_t i = 0; i < slots; ++i) {
        if (pc[i] == 1.0 && active == slots) {
            active = i;
        } else if (pc[i] != 0.0) {
            return 1;
        }
    }
    if (active == slots || targets[2 * active] < 0) {
        return 1;
    }

    double zero = 1.0;
    for (int i = 0; i < width; ++i) {
        if (condition[i] != 0.0 && condition[i] != 1.0) {
            return 1;
        }
        zero = and_gate(zero, 1.0 - condition[i], and_weight);
    }
    const double nonzero = 1.0 - zero;
    const double* flags = guards + 3 * active;
    const double take = guard_weights[0] * flags[0]
                      + guard_weights[1] * flags[1] * nonzero
                      + guard_weights[2] * flags[2] * zero;
    const double normal_gain = path_weights[0] * (1.0 - take);
    const double branch_gain = path_weights[1] * take;
    if (!std::isfinite(normal_gain) || !std::isfinite(branch_gain)) {
        return 2;
    }

    const auto normal = targets[2 * active];
    const auto alternate = targets[2 * active + 1];
    auto next = normal;
    // Two nonzero coordinates may coincide. Combine their gains before
    // admitting the output, exactly as in the dense bilinear contraction.
    if (normal == alternate) {
        if (normal_gain + branch_gain != 1.0) {
            return 2;
        }
    } else if (normal_gain == 0.0 && branch_gain == 1.0) {
        next = alternate;
    } else if (normal_gain != 1.0 || branch_gain != 0.0) {
        return 2;
    }

    details[0] = static_cast<double>(next);
    details[1] = nonzero;
    details[2] = take;
    details[3] = static_cast<double>(normal);
    details[4] = static_cast<double>(alternate);
    return 0;
}
