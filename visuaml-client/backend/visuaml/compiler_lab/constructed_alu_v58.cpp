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

extern "C" int visuaml_indexed_route(
        std::size_t active, std::size_t slots, const double* condition,
        const std::int32_t* targets, const double* guards,
        const double* guard_weights, const double* path_weights,
        const double* and_weight, double* details) {
    if (active >= slots || targets[2 * active] < 0) {
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
    return visuaml_indexed_route(active, slots, condition, targets, guards,
                                  guard_weights, path_weights, and_weight, details);
}

namespace {
constexpr int alu_count = 6;
constexpr int carry_count = 32;

// Layout of prepared float64 rows and fixed coefficients.
enum Row { retain, push, pop, unary, binary, no_local, get, set, tee, alu, immediate = alu + alu_count };
enum Gain { pop_first, pop_second, produce_const, produce_local, produce_alu, write_local,
            actions, output = actions + 4, and_kernel = output + alu_count,
            xor_kernel = and_kernel + 3, carry = xor_kernel + 4, adder = carry + carry_count };

enum Status {
    ok, operand_binary, operand_onehot, inactive_cells, underflow, overflow,
    micro_binary, alu_binary, alu_selection, local_onehot, local_binary,
    local_read, local_write, stack_authority,
};

bool binary_tensor(const double* values, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] != 0.0 && values[i] != 1.0) {
            return false;
        }
    }
    return true;
}

double sum(const double* values, std::size_t count) {
    double value = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        value += values[i];
    }
    return value;
}
} // namespace

extern "C" int visuaml_data_microstep(
        std::size_t pc, std::size_t stack_size, std::size_t locals_size,
        const double* rows, const double* fixed, double* height, double* cells, double* locals) {
    // The prepared caller owns the buffers. Check the initial state in the
    // same order as the fused NumPy equations before reading its active slots.
    if (!binary_tensor(height, stack_size + 1)) {
        return operand_binary;
    }
    if (sum(height, stack_size + 1) != 1.0) {
        return operand_onehot;
    }
    std::size_t depth = 0;
    while (height[depth] == 0.0) {
        ++depth;
    }
    if (!binary_tensor(cells, stack_size * width)) {
        return operand_binary;
    }
    for (std::size_t i = depth * width; i < stack_size * width; ++i) {
        if (cells[i] != 0.0) {
            return inactive_cells;
        }
    }
    if (!binary_tensor(locals, locals_size * width) || !binary_tensor(fixed, actions)) {
        return micro_binary;
    }
    const double* action = fixed + actions;
    if (!binary_tensor(action, 4)) {
        return operand_binary;
    }

    const double* row = rows + pc * (immediate + width + locals_size + 1);
    const double* local_selector = row + immediate + width;
    const double const_mask = local_selector[locals_size];
    const double first = (row[pop] + row[unary] + row[binary]) * fixed[pop_first];
    const double second = row[binary] * fixed[pop_second];
    if (first && depth == 0) {
        return underflow;
    }
    if ((1.0 - first) * action[0] + first * action[2] != 1.0) {
        return operand_onehot;
    }
    const auto after_first = depth - static_cast<std::size_t>(first);
    if (second && after_first == 0) {
        return underflow;
    }
    if ((1.0 - second) * action[0] + second * action[2] != 1.0) {
        return operand_onehot;
    }
    const auto after_second = after_first - static_cast<std::size_t>(second);
    const double zero[width] = {};
    const double* top = depth ? cells + (depth - 1) * width : zero;
    const double* previous = after_first ? cells + (after_first - 1) * width : zero;

    double left[width], right[width];
    for (int i = 0; i < width; ++i) {
        left[i] = row[unary] * first * top[i] + row[binary] * second * previous[i];
        right[i] = row[binary] * first * top[i];
    }
    if (!binary_tensor(left, width) || !binary_tensor(right, width)) {
        return micro_binary;
    }
    if (!binary_tensor(fixed + output, alu_count)) {
        return alu_binary;
    }

    // Even for non-ALU rows, evaluate the same six bit-gate candidates
    // and the default add selector used by the Python reference.
    double candidates[alu_count * width] = {};
    visuaml_alu_candidates(left, right, fixed + and_kernel, fixed + xor_kernel,
                           fixed + carry, carry_count, fixed + adder, candidates);
    if (!binary_tensor(candidates, alu_count * width)) {
        return alu_binary;
    }
    const double is_alu = sum(row + alu, alu_count);
    double alu_gate[alu_count];
    for (int i = 0; i < alu_count; ++i) {
        alu_gate[i] = (row[alu + i] + (i == 0 ? 1.0 - is_alu : 0.0)) * fixed[output + i];
    }
    if (sum(alu_gate, alu_count) != 1.0) {
        return alu_selection;
    }

    // Select one coordinate from the exact one-hot read and write
    // contractions. Both gates are live even when their result is masked.
    std::size_t slot = locals_size;
    for (std::size_t i = 0; i < locals_size; ++i) {
        const double selected = local_selector[i] + (i == 0 ? row[no_local] : 0.0);
        if (selected == 1.0 && slot == locals_size) {
            slot = i;
        } else if (selected != 0.0) {
            return local_onehot;
        }
    }
    if (slot == locals_size) {
        return local_onehot;
    }
    const double* read_gain = fixed + adder + 2;
    const double* write_gain = read_gain + locals_size;
    if (!binary_tensor(read_gain, locals_size)) {
        return local_binary;
    }
    if (read_gain[slot] != 1.0) {
        return local_read;
    }
    if (!binary_tensor(write_gain, locals_size)) {
        return local_binary;
    }
    if (write_gain[slot] != 1.0) {
        return local_write;
    }
    const double local_mask = (row[set] + row[tee]) * fixed[write_local];
    double bank_word[width];
    double* local = locals + slot * width;
    for (int i = 0; i < width; ++i) {
        const double value = row[set] * first * top[i] + row[tee] * top[i];
        bank_word[i] = (1.0 - local_mask) * local[i] + local_mask * value;
    }

    const double sources[] = {
        const_mask * fixed[produce_const], row[get] * fixed[produce_local],
        is_alu * fixed[produce_alu],
    };
    double value[width];
    for (int i = 0; i < width; ++i) {
        double alu_value = 0.0;
        for (int j = 0; j < alu_count; ++j) {
            alu_value += alu_gate[j] * candidates[j * width + i];
        }
        value[i] = sources[0] * row[immediate + i] + sources[1] * local[i] + sources[2] * alu_value;
    }
    if (!binary_tensor(value, width) || !binary_tensor(bank_word, width)) {
        return micro_binary;
    }
    const double push_expected = row[push] + row[unary] + row[binary];
    if (sum(sources, 3) != push_expected || first + second != row[pop] + row[unary] + 2.0 * row[binary]) {
        return stack_authority;
    }
    if (push_expected && after_second == stack_size) {
        return overflow;
    }
    if ((1.0 - push_expected) * action[0] + push_expected * action[1] != 1.0) {
        return operand_onehot;
    }

    // All validation precedes state updates. The contracted retain/pop
    // gates clear precisely the slots that left the active prefix.
    const auto next_depth = after_second + static_cast<std::size_t>(push_expected);
    for (std::size_t i = next_depth * width; i < depth * width; ++i) {
        cells[i] = 0.0;
    }
    if (push_expected) {
        for (int i = 0; i < width; ++i) {
            cells[after_second * width + i] = value[i];
        }
    }
    for (int i = 0; i < width; ++i) {
        local[i] = bank_word[i];
    }
    height[depth] = 0.0;
    height[next_depth] = 1.0;
    return ok;
}
