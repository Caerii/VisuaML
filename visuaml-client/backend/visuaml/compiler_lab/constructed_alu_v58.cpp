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
    local_read, local_write, stack_authority, route_binary, route_selection,
    early_terminal, final_stack, fuel_exhausted, trace_capacity,
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
// The caller supplies an admitted active prefix and its depth. The
// transition preserves that invariant on every successful microstep.
int data_microstep_at_depth(
        std::size_t pc, std::size_t stack_size, std::size_t locals_size,
        const double* rows, const double* fixed, std::size_t& depth, double* cells, double* locals) {
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
    depth = next_depth;
    return ok;
}


// Keep the signed i32 representation independent of implementation-defined
// signed conversions when the high bit is set.
std::int32_t decode_word(const double* bits) {
    std::uint32_t word = 0;
    for (int i = 0; i < width; ++i) {
        word |= static_cast<std::uint32_t>(bits[i]) << i;
    }
    const auto signed_word = static_cast<std::int64_t>(word)
        - (static_cast<std::int64_t>(word >> 31) << 32);
    return static_cast<std::int32_t>(signed_word);
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
    const auto old_depth = depth;
    const auto status = data_microstep_at_depth(pc, stack_size, locals_size, rows, fixed, depth, cells, locals);
    if (status == ok) {
        height[old_depth] = 0.0;
        height[depth] = 1.0;
    }
    return status;
}

namespace {
constexpr std::size_t prepared_stack_size = 24;
constexpr std::size_t prepared_locals_size = 5;

int prepared_loop(
        std::int32_t argument, std::size_t fuel, std::size_t count, std::size_t terminal,
        std::size_t visible_locals, const double* rows, const double* fixed,
        const std::int32_t* targets, const double* guards, const double* guard_weights,
        const double* path_weights, const double* and_weight, std::int32_t* result,
        std::int32_t* trace_states, double* trace_routes) {
    constexpr auto stack_size = prepared_stack_size;
    constexpr auto locals_size = prepared_locals_size;
    double cells[stack_size * width] = {};
    double locals[locals_size * width] = {};
    const double zero[width] = {};
    const auto input = static_cast<std::uint32_t>(argument);
    for (int i = 0; i < width; ++i) {
        locals[i] = (input >> i) & 1U;
    }
    std::size_t pc = 0;
    std::size_t depth = 0;
    for (std::size_t tick = 0; tick < fuel; ++tick) {
        if (pc >= count || pc == terminal) {
            return early_terminal;
        }
        const double* top = depth ? cells + (depth - 1) * width : zero;
        double route[5];
        const auto route_status = visuaml_indexed_route(
            pc, terminal + 1, top, targets, guards, guard_weights, path_weights, and_weight, route);
        if (route_status) {
            return route_status == 1 ? route_binary : route_selection;
        }
        const auto next = static_cast<std::size_t>(route[0]);
        const bool halted = pc == count - 1;
        if (halted) {
            if (depth != 1 || next != terminal) {
                return final_stack;
            }
        } else {
            const auto status = data_microstep_at_depth(
                pc, stack_size, locals_size, rows, fixed, depth, cells, locals);
            if (status) {
                return status;
            }
        }
        if (trace_states) {
            // The caller reserves min(fuel, count) rows for straight-line WASM.
            if (tick >= count) {
                return trace_capacity;
            }
            auto* state = trace_states + tick * (2 + stack_size + visible_locals);
            state[0] = static_cast<std::int32_t>(pc);
            state[1] = static_cast<std::int32_t>(depth);
            for (std::size_t slot = 0; slot < depth; ++slot) {
                state[2 + slot] = decode_word(cells + slot * width);
            }
            for (std::size_t slot = 0; slot < visible_locals; ++slot) {
                state[2 + stack_size + slot] = decode_word(locals + slot * width);
            }
            for (int field = 0; field < 5; ++field) {
                trace_routes[tick * 5 + field] = route[field];
            }
        }
        if (halted) {
            result[0] = decode_word(cells);
            result[1] = static_cast<std::int32_t>(tick + 1);
            return ok;
        }
        pc = next;
    }
    return fuel_exhausted;
}

} // namespace

extern "C" int visuaml_prepared_loop(
        std::int32_t argument, std::size_t fuel, std::size_t count, std::size_t terminal,
        std::size_t visible_locals, const double* rows, const double* fixed,
        const std::int32_t* targets, const double* guards, const double* guard_weights,
        const double* path_weights, const double* and_weight, std::int32_t* result,
        std::int32_t* trace_states, double* trace_routes) {
    return prepared_loop(argument, fuel, count, terminal, visible_locals, rows, fixed,
                         targets, guards, guard_weights, path_weights, and_weight,
                         result, trace_states, trace_routes);
}

extern "C" void visuaml_prepared_batch(
        const std::int32_t* arguments, const std::int32_t* fuels, std::size_t size,
        std::size_t count, std::size_t terminal, std::size_t visible_locals, std::size_t trace_stride,
        const double* rows, const double* fixed, const std::int32_t* targets, const double* guards,
        const double* guard_weights, const double* path_weights, const double* and_weight,
        std::int32_t* results, std::int32_t* statuses, std::int32_t* trace_states, double* trace_routes) {
    const auto state_stride = trace_stride * (2 + prepared_stack_size + visible_locals);
    const auto route_stride = trace_stride * 5;
    for (std::size_t i = 0; i < size; ++i) {
        // Loop state is invocation-local, including when a neighboring input traps.
        statuses[i] = prepared_loop(
            arguments[i], fuels[i], count, terminal, visible_locals, rows, fixed,
            targets, guards, guard_weights, path_weights, and_weight, results + 2 * i,
            trace_states ? trace_states + i * state_stride : nullptr,
            trace_routes ? trace_routes + i * route_stride : nullptr);
    }
}
