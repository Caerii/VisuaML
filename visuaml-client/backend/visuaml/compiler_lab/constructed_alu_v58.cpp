#include <cstdint>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <type_traits>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define VISUAML_AVX2 1
#define VISUAML_INLINE inline __attribute__((always_inline))
#else
#define VISUAML_AVX2 0
#define VISUAML_INLINE inline
#endif

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

// v54 event graph, after contracting the source opcode with the fixed
// default matrix and conditional gains. No WASM state or PC is accepted.
extern "C" int visuaml_frame_event_gate(
        const double* row, bool sequential, bool has_else,
        bool frame_active, bool branch_taken, double* output) {
    const double both = static_cast<double>(sequential) + static_cast<double>(has_else);
    const double enter_if = both - relu(both - 1.0);
    const double push = row[5] * enter_if;
    const double pop = row[6] * static_cast<double>(frame_active);
    const double branch = row[7] * static_cast<double>(branch_taken);
    output[0] = row[0] - (push + pop + branch);
    output[1] = row[1] + push;
    output[2] = row[2] + pop;
    output[3] = row[3] + branch;
    output[4] = row[4];
    int selected = -1;
    for (int i = 0; i < 5; ++i) {
        if (output[i] == 1.0 && selected < 0) {
            selected = i;
        } else if (output[i] != 0.0) {
            return -1;
        }
    }
    return selected;
}

// The v53 unit-retain contraction is the identity on the three frame
// tensors. The copying arm materializes that identity in a second buffer.
// An absent destination retains the private input representation in place.
extern "C" int visuaml_frame_event_retain(
        const double* row, bool sequential, bool has_else, bool frame_active,
        bool branch_taken, double* event, const double* state, double* destination,
        std::size_t elements) {
    const int action = visuaml_frame_event_gate(
        row, sequential, has_else, frame_active, branch_taken, event);
    if (action == 0 && destination != nullptr) {
        std::memcpy(destination, state, elements * sizeof(double));
    }
    return action;
}

// The lexical else bit travels with the prepared v54 coefficients.
extern "C" int visuaml_prepared_frame_event_retain(
        const double* row, bool sequential, bool frame_active, bool branch_taken,
        double* event, const double* state, double* destination, std::size_t elements) {
    return visuaml_frame_event_retain(row, sequential, row[8] == 1.0, frame_active,
                                     branch_taken, event, state, destination, elements);
}

namespace {
int route_from_zero(
        std::size_t active, double zero, const std::int32_t* targets,
        const double* guards, const double* guard_weights, const double* path_weights,
        double* details) {
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

} // namespace

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
    return route_from_zero(active, zero, targets, guards, guard_weights, path_weights, details);
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
    control_effect, control_frame, control_event, frame_gain, invalid_return, memory_oob,
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
// Canonical carry reduction gives arithmetic modulo 2^32. Flipping the
// sign bit maps signed two's-complement ordering to unsigned ordering.
VISUAML_INLINE void packed_candidates(std::uint32_t left, std::uint32_t right, std::uint32_t* values) {
    constexpr std::uint32_t sign = std::uint32_t{1} << 31;
    values[0] = left + right;
    values[1] = left - right;
    values[2] = left * right;
    values[3] = (left ^ sign) < (right ^ sign);
    values[4] = (left ^ sign) > (right ^ sign);
    values[5] = left == 0;
}

std::uint32_t masked_word(std::uint32_t word, double bit) {
    return word & (std::uint32_t{0} - static_cast<std::uint32_t>(bit));
}

bool any_word(std::uint32_t word) {
    return word != 0;
}

#if VISUAML_AVX2
constexpr std::size_t lane_count = 8;
struct LaneWord {
    std::uint32_t values[lane_count]{};

    LaneWord() = default;
    explicit LaneWord(std::uint32_t value) {
        for (auto& lane : values) {
            lane = value;
        }
    }
};

VISUAML_INLINE LaneWord operator|(const LaneWord& left, const LaneWord& right) {
    LaneWord result;
    for (std::size_t i = 0; i < lane_count; ++i) {
        result.values[i] = left.values[i] | right.values[i];
    }
    return result;
}

VISUAML_INLINE LaneWord& operator|=(LaneWord& left, const LaneWord& right) {
    return left = left | right;
}

VISUAML_INLINE LaneWord operator&(const LaneWord& left, const LaneWord& right) {
    LaneWord result;
    for (std::size_t i = 0; i < lane_count; ++i) {
        result.values[i] = left.values[i] & right.values[i];
    }
    return result;
}

VISUAML_INLINE LaneWord masked_word(const LaneWord& word, double bit) {
    return word & LaneWord(std::uint32_t{0} - static_cast<std::uint32_t>(bit));
}

VISUAML_INLINE bool any_word(const LaneWord& word) {
    std::uint32_t combined = 0;
    for (auto lane : word.values) {
        combined |= lane;
    }
    return combined != 0;
}

VISUAML_INLINE void packed_candidates(const LaneWord& left, const LaneWord& right, LaneWord* out) {
    for (std::size_t i = 0; i < lane_count; ++i) {
        std::uint32_t candidates[alu_count];
        packed_candidates(left.values[i], right.values[i], candidates);
        for (int candidate = 0; candidate < alu_count; ++candidate) {
            out[candidate].values[i] = candidates[candidate];
        }
    }
}
#endif

template <typename Word>
constexpr bool is_packed = !std::is_same_v<Word, double>;

template <typename Word>
constexpr std::size_t word_stride = is_packed<Word> ? 1 : width;

// The caller supplies an admitted active prefix and its depth. The
// transition preserves that invariant on every successful microstep.
template <typename Word>
VISUAML_INLINE int data_microstep_at_depth(
        std::size_t pc, std::size_t stack_size, std::size_t locals_size,
        const double* rows, const double* fixed, std::size_t& depth, Word* cells, Word* locals) {
    constexpr auto stride = word_stride<Word>;
    if constexpr (!is_packed<Word>) {
        if (!binary_tensor(locals, locals_size * stride)) {
            return micro_binary;
        }
    }
    if (!binary_tensor(fixed, actions)) {
        return micro_binary;
    }
    const double* action = fixed + actions;
    if (!binary_tensor(action, 4)) {
        return operand_binary;
    }

    const double* row = rows + pc * (immediate + stride + locals_size + 1);
    const double* local_selector = row + immediate + stride;
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
    const Word zero[stride] = {};
    const Word* top = depth ? cells + (depth - 1) * stride : zero;
    const Word* previous = after_first ? cells + (after_first - 1) * stride : zero;

    // Non-ALU rows also evaluate the six candidates and default add gate.
    Word candidates[alu_count * stride] = {};
    if constexpr (is_packed<Word>) {
        if (!binary_tensor(fixed + output, alu_count)) {
            return alu_binary;
        }
        const auto left = masked_word(*top, row[unary] * first)
            | masked_word(*previous, row[binary] * second);
        const auto right = masked_word(*top, row[binary] * first);
        packed_candidates(left, right, candidates);
    } else {
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
        visuaml_alu_candidates(left, right, fixed + and_kernel, fixed + xor_kernel,
                               fixed + carry, carry_count, fixed + adder, candidates);
        if (!binary_tensor(candidates, alu_count * width)) {
            return alu_binary;
        }
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
    Word bank_word[stride], value[stride];
    Word* local = locals + slot * stride;
    const double sources[] = {
        const_mask * fixed[produce_const], row[get] * fixed[produce_local],
        is_alu * fixed[produce_alu],
    };
    if constexpr (is_packed<Word>) {
        const auto write_value = masked_word(*top, row[set] * first + row[tee]);
        bank_word[0] = masked_word(*local, 1.0 - local_mask) | masked_word(write_value, local_mask);
        Word alu_value{};
        for (int j = 0; j < alu_count; ++j) {
            alu_value |= masked_word(candidates[j], alu_gate[j]);
        }
        const auto literal = masked_word(static_cast<Word>(static_cast<std::uint32_t>(row[immediate])), sources[0]);
        const auto read = masked_word(*local, sources[1]);
        const auto computed = masked_word(alu_value, sources[2]);
        // Independent source gates may overlap under an altered decoder.
        // Preserve the float path's binary-output trap before its arity check.
        if (any_word((literal & read) | ((literal | read) & computed))) {
            return micro_binary;
        }
        value[0] = literal | read | computed;
    } else {
        for (int i = 0; i < width; ++i) {
            const double write_value = row[set] * first * top[i] + row[tee] * top[i];
            bank_word[i] = (1.0 - local_mask) * local[i] + local_mask * write_value;
            double alu_value = 0.0;
            for (int j = 0; j < alu_count; ++j) {
                alu_value += alu_gate[j] * candidates[j * width + i];
            }
            value[i] = sources[0] * row[immediate + i] + sources[1] * local[i] + sources[2] * alu_value;
        }
        if (!binary_tensor(value, width) || !binary_tensor(bank_word, width)) {
            return micro_binary;
        }
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
    for (std::size_t i = next_depth * stride; i < depth * stride; ++i) {
        cells[i] = Word{};
    }
    if (push_expected) {
        for (std::size_t i = 0; i < stride; ++i) {
            cells[after_second * stride + i] = value[i];
        }
    }
    for (std::size_t i = 0; i < stride; ++i) {
        local[i] = bank_word[i];
    }
    depth = next_depth;
    return ok;
}


// Keep the signed i32 representation independent of implementation-defined
// signed conversions when the high bit is set.
std::int32_t signed_word(std::uint32_t word) {
    const auto converted = static_cast<std::int64_t>(word)
        - (static_cast<std::int64_t>(word >> 31) << 32);
    return static_cast<std::int32_t>(converted);
}

std::int32_t decode_word(const std::uint32_t* value) {
    return signed_word(*value);
}

std::int32_t decode_word(const double* bits) {
    std::uint32_t word = 0;
    for (int i = 0; i < width; ++i) {
        word |= static_cast<std::uint32_t>(bits[i]) << i;
    }
    return signed_word(word);
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
constexpr std::size_t prepared_frame_limit = 12;

template <typename Word>
void record_state(std::int32_t* states, double* routes, std::size_t tick,
                  std::size_t pc, std::size_t depth, std::size_t visible_locals,
                  const Word* cells, const Word* locals, const double* route) {
    constexpr auto stride = word_stride<Word>;
    auto state = states + tick * (2 + prepared_stack_size + visible_locals);
    state[0] = static_cast<std::int32_t>(pc);
    state[1] = static_cast<std::int32_t>(depth);
    for (std::size_t slot = 0; slot < depth; ++slot) {
        state[2 + slot] = decode_word(cells + slot * stride);
    }
    for (std::size_t slot = 0; slot < visible_locals; ++slot) {
        state[2 + prepared_stack_size + slot] = decode_word(locals + slot * stride);
    }
    for (int field = 0; field < 5; ++field) {
        routes[tick * 5 + field] = route[field];
    }
}

template <typename Word>
int prepared_loop(
        std::int32_t argument, std::size_t fuel, std::size_t count, std::size_t terminal,
        std::size_t visible_locals, const double* rows, const double* fixed,
        const std::int32_t* targets, const double* guards, const double* guard_weights,
        const double* path_weights, const double* and_weight, std::int32_t* result,
        std::int32_t* trace_states, double* trace_routes) {
    constexpr auto stack_size = prepared_stack_size;
    constexpr auto locals_size = prepared_locals_size;
    constexpr auto stride = word_stride<Word>;
    Word cells[stack_size * stride] = {};
    Word locals[locals_size * stride] = {};
    const Word zero[stride] = {};
    const auto input = static_cast<std::uint32_t>(argument);
    if constexpr (is_packed<Word>) {
        locals[0] = input;
    } else {
        for (int i = 0; i < width; ++i) {
            locals[i] = (input >> i) & 1U;
        }
    }
    std::size_t pc = 0;
    std::size_t depth = 0;
    for (std::size_t tick = 0; tick < fuel; ++tick) {
        if (pc >= count || pc == terminal) {
            return early_terminal;
        }
        const Word* top = depth ? cells + (depth - 1) * stride : zero;
        double route[5];
        int route_status;
        if constexpr (is_packed<Word>) {
            route_status = targets[2 * pc] < 0 ? 1 : route_from_zero(
                pc, *top == 0, targets, guards, guard_weights, path_weights, route);
        } else {
            route_status = visuaml_indexed_route(
                pc, terminal + 1, top, targets, guards, guard_weights, path_weights, and_weight, route);
        }
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
            record_state(trace_states, trace_routes, tick, pc, depth, visible_locals, cells, locals, route);
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

#if VISUAML_AVX2
__attribute__((target("avx2")))
bool prepared_avx2(
        const std::int32_t* arguments, const std::int32_t* fuels,
        std::size_t count, std::size_t terminal, const double* rows, const double* fixed,
        const std::int32_t* targets, const double* guards, const double* guard_weights,
        const double* path_weights, std::int32_t* result, std::int32_t* statuses) {
    for (std::size_t i = 0; i < lane_count; ++i) {
        if (static_cast<std::size_t>(fuels[i]) < count) {
            return false;
        }
    }
    LaneWord cells[prepared_stack_size], locals[prepared_locals_size];
    for (std::size_t i = 0; i < lane_count; ++i) {
        locals[0].values[i] = static_cast<std::uint32_t>(arguments[i]);
    }
    std::size_t depth = 0, pc = 0;
    for (std::size_t tick = 0; tick < count; ++tick) {
        if (pc >= count || pc == terminal || targets[2 * pc] < 0) {
            return false;
        }
        const auto flags = guards + 3 * pc;
        // Straight-line rows are independent of the input condition. Keep
        // live route-gain evaluation, and replay unsupported rows scalar.
        if (flags[1] != 0.0 || flags[2] != 0.0) {
            return false;
        }
        double route[5];
        if (route_from_zero(pc, 0.0, targets, guards, guard_weights, path_weights, route)) {
            return false;
        }
        const auto next = static_cast<std::size_t>(route[0]);
        if (pc == count - 1) {
            if (depth != 1 || next != terminal) {
                return false;
            }
            for (std::size_t i = 0; i < lane_count; ++i) {
                result[2 * i] = signed_word(cells[0].values[i]);
                result[2 * i + 1] = static_cast<std::int32_t>(tick + 1);
                statuses[i] = ok;
            }
            return true;
        }
        if (data_microstep_at_depth(pc, prepared_stack_size, prepared_locals_size,
                                    rows, fixed, depth, cells, locals)) {
            return false;
        }
        pc = next;
    }
    return false;
}
#endif

} // namespace

extern "C" int visuaml_prepared_loop(
        std::int32_t argument, std::size_t fuel, std::size_t count, std::size_t terminal,
        std::size_t visible_locals, const double* rows, const double* fixed,
        const std::int32_t* targets, const double* guards, const double* guard_weights,
        const double* path_weights, const double* and_weight, std::int32_t* result,
        std::int32_t* trace_states, double* trace_routes, bool packed) {
    auto loop = packed ? prepared_loop<std::uint32_t> : prepared_loop<double>;
    return loop(argument, fuel, count, terminal, visible_locals, rows, fixed,
                         targets, guards, guard_weights, path_weights, and_weight,
                         result, trace_states, trace_routes);
}

extern "C" bool visuaml_prepared_simd_available() {
#if VISUAML_AVX2
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}

extern "C" void visuaml_prepared_batch(
        const std::int32_t* arguments, const std::int32_t* fuels, std::size_t size,
        std::size_t count, std::size_t terminal, std::size_t visible_locals, std::size_t trace_stride,
        const double* rows, const double* fixed, const std::int32_t* targets, const double* guards,
        const double* guard_weights, const double* path_weights, const double* and_weight,
        std::int32_t* results, std::int32_t* statuses, std::int32_t* trace_states, double* trace_routes, bool packed, bool simd) {
    auto loop = packed ? prepared_loop<std::uint32_t> : prepared_loop<double>;
    const auto state_stride = trace_stride * (2 + prepared_stack_size + visible_locals);
    const auto route_stride = trace_stride * 5;
    std::size_t i = 0;
    const auto scalar = [&](std::size_t i) {
        // Loop state is invocation-local, including when a neighboring input traps.
        statuses[i] = loop(
            arguments[i], fuels[i], count, terminal, visible_locals, rows, fixed,
            targets, guards, guard_weights, path_weights, and_weight, results + 2 * i,
            trace_states ? trace_states + i * state_stride : nullptr,
            trace_routes ? trace_routes + i * route_stride : nullptr);
    };
#if VISUAML_AVX2
    if (simd && packed && !trace_states && visuaml_prepared_simd_available()) {
        for (; size - i >= lane_count; i += lane_count) {
            if (!prepared_avx2(arguments + i, fuels + i, count, terminal, rows, fixed,
                               targets, guards, guard_weights, path_weights,
                               results + 2 * i, statuses + i)) {
                for (std::size_t lane = 0; lane < lane_count; ++lane) {
                    scalar(i + lane);
                }
            }
        }
    }
#else
    (void)simd;
#endif
    for (; i < size; ++i) {
        scalar(i);
    }
}

namespace {
// Coordinates from v54 event tokens and v53 one-hot frame states.
enum ControlOp { block_op, loop_op, if_op, else_op, end_op, br_op, br_if_op, return_op, data_op, load_op, store_op };
enum FrameAction { keep_frame, push_frame, pop_frame, jump_frame, return_frame };
constexpr int frame_actions = 5;
constexpr int control_fields = 6;
constexpr int frame_fields = 2 + 2 * prepared_frame_limit;
constexpr int memory_fields = 3;
constexpr std::uint64_t memory_bytes = 65536;

int memory_step(int opcode, const std::int64_t* instruction, const double* action,
                std::uint32_t* cells, std::size_t& depth, std::uint8_t* memory,
                const std::uint8_t* gains, std::uint32_t* trace, std::int64_t* attempted) {
    const bool store = opcode == store_op;
    const std::size_t pops = store ? 2 : 1;
    if (instruction[4] != static_cast<std::int64_t>(pops)) {
        return control_effect;
    }
    if (depth < pops) {
        return underflow;
    }
    if (!binary_tensor(action, 4)) {
        return operand_binary;
    }
    if (action[2] != 1.0 || (!store && action[1] != 1.0)) {
        return operand_onehot;
    }
    // WASM memory32 uses an unsigned input address plus an unsigned offset
    // in a widened sum. Admission precedes all byte reads and writes.
    const auto address = std::uint64_t{cells[depth - pops]} + instruction[5];
    if (!memory || address > memory_bytes - 4) {
        if (attempted) {
            *attempted = static_cast<std::int64_t>(address);
        }
        return memory_oob;
    }
    auto word = memory + address;
    std::uint32_t before = 0;
    for (int i = 0; i < 4; ++i) {
        before |= std::uint32_t{word[i]} << (8 * i);
    }
    auto after = before;
    if (store) {
        if (gains[1]) {
            after = cells[depth - 1];
            for (int i = 0; i < 4; ++i) {
                word[i] = static_cast<std::uint8_t>(after >> (8 * i));
            }
        }
        cells[--depth] = 0;
        cells[--depth] = 0;
    } else {
        cells[depth - 1] = gains[0] ? before : 0;
    }
    if (trace) {
        trace[0] = static_cast<std::uint32_t>(address);
        trace[1] = before;
        trace[2] = after;
    }
    return ok;
}

struct ActiveFrame {
    std::size_t pc;
    std::size_t base;
};

int select_frame_event(const double* defaults, const double* gains, int opcode,
                       bool has_else, bool sequential, bool active, bool taken) {
    const double enter = (opcode == if_op && (has_else || sequential)) ? gains[0] : 0.0;
    const double leave = (opcode == end_op && active) ? gains[1] : 0.0;
    const double branch = (opcode == br_if_op && taken) ? gains[2] : 0.0;
    const double shifts[] = {-(enter + leave + branch), enter, leave, branch, 0.0};
    int action = -1;
    const auto row = defaults + opcode * frame_actions;
    for (int i = 0; i < frame_actions; ++i) {
        const auto selected = row[i] + shifts[i];
        if (selected == 1.0 && action < 0) {
            action = i;
        } else if (selected != 0.0) {
            return -1;
        }
    }
    return action;
}

int structured_loop(
        std::int32_t argument, std::size_t fuel, std::size_t count, std::size_t terminal,
        std::size_t visible_locals, const double* rows, const double* fixed,
        const std::int32_t* targets, const double* guards, const double* guard_weights,
        const double* path_weights, const std::int64_t* flow, const double* event_defaults,
        const double* event_gains, const double* frame_gains, std::int32_t* result,
        std::int32_t* trace_states, double* trace_routes, std::int32_t* trace_frames,
        std::uint8_t* memory, const std::uint8_t* memory_gains, std::uint32_t* trace_memory, std::int64_t* trap) {
    std::uint32_t cells[prepared_stack_size] = {}, locals[prepared_locals_size] = {};
    ActiveFrame frames[prepared_frame_limit];
    locals[0] = static_cast<std::uint32_t>(argument);
    std::size_t pc = 0, depth = 0, frame_count = 0;
    for (std::size_t tick = 0; tick < fuel; ++tick) {
        if (trap) {
            trap[0] = static_cast<std::int64_t>(tick);
            trap[1] = static_cast<std::int64_t>(pc);
        }
        if (pc >= count || pc == terminal) {
            return early_terminal;
        }
        if (targets[2 * pc] < 0) {
            return route_binary;
        }
        double route[5];
        const bool zero = depth == 0 || cells[depth - 1] == 0;
        if (route_from_zero(pc, zero, targets, guards, guard_weights, path_weights, route)) {
            return route_selection;
        }
        const auto next = static_cast<std::size_t>(route[0]);
        const auto instruction = flow + control_fields * pc;
        const int opcode = instruction[0];
        bool halted = false;
        if (opcode == data_op) {
            const auto status = data_microstep_at_depth(
                pc, prepared_stack_size, prepared_locals_size, rows, fixed, depth, cells, locals);
            if (status) {
                return status;
            }
        } else if (opcode == load_op || opcode == store_op) {
            auto access = trace_memory ? trace_memory + tick * memory_fields : nullptr;
            const auto status = memory_step(opcode, instruction, fixed + actions, cells, depth,
                                            memory, memory_gains, access, trap ? trap + 2 : nullptr);
            if (status) {
                return status;
            }
        } else {
            // The source is an admission oracle for the decoded control arity.
            // Route selection above reads the actual operand before consumption.
            const int pops = instruction[4];
            if (pops != (opcode == if_op || opcode == br_if_op ? 1 : 0)) {
                return control_effect;
            }
            if (pops) {
                if (!depth) {
                    return underflow;
                }
                if (!binary_tensor(fixed + actions, 4)) {
                    return operand_binary;
                }
                if (fixed[actions + 2] != 1.0) {
                    return operand_onehot;
                }
                cells[--depth] = 0;
            }
            if (opcode == else_op || opcode == end_op) {
                if (frame_count) {
                    const auto& top = frames[frame_count - 1];
                    const auto matching = flow + control_fields * top.pc;
                    if (matching[opcode == else_op ? 2 : 1] != static_cast<std::int32_t>(pc)
                            || depth != top.base) {
                        return control_frame;
                    }
                } else if (opcode == end_op && pc == count - 1 && depth == 1) {
                    halted = true;
                } else {
                    return control_frame;
                }
            } else if (opcode == return_op) {
                if (depth != 1 || next != terminal) {
                    return invalid_return;
                }
                halted = true;
            }
        }

        const int action = select_frame_event(
            event_defaults, event_gains, opcode > data_op ? data_op : opcode, instruction[2] >= 0,
            next == route[3], frame_count != 0, route[2] != 0.0);
        if (action < 0) {
            return control_event;
        }
        if (frame_gains[action] != 1.0) {
            return frame_gain;
        }
        // The live event gate selects the contraction. Lexical metadata
        // supplies frame tags and the packed coordinates of their bases.
        if (action == push_frame) {
            if (instruction[1] < 0 || frame_count == prepared_frame_limit) {
                return control_frame;
            }
            frames[frame_count++] = {pc, depth};
        } else if (action == pop_frame) {
            if (!frame_count) {
                return control_frame;
            }
            --frame_count;
        } else if (action == jump_frame) {
            const auto branch_depth = static_cast<std::size_t>(instruction[3]);
            if (branch_depth >= frame_count) {
                return control_frame;
            }
            const auto index = frame_count - 1 - branch_depth;
            const auto target = frames[index];
            if (target.base > depth) {
                return control_frame;
            }
            if (!binary_tensor(fixed + actions, 4)) {
                return operand_binary;
            }
            if (fixed[actions + 3] != 1.0) {
                return operand_onehot;
            }
            while (depth > target.base) {
                cells[--depth] = 0;
            }
            frame_count = index + (flow[control_fields * target.pc] == loop_op);
        } else if (action == return_frame) {
            frame_count = 0;
        }
        if (halted && next != terminal) {
            return invalid_return;
        }
        if (trace_states) {
            record_state(trace_states, trace_routes, tick, pc, depth, visible_locals, cells, locals, route);
            auto frame = trace_frames + tick * frame_fields;
            frame[0] = action;
            frame[1] = static_cast<std::int32_t>(frame_count);
            for (std::size_t i = 0; i < frame_count; ++i) {
                frame[2 + 2 * i] = static_cast<std::int32_t>(frames[i].pc);
                frame[3 + 2 * i] = static_cast<std::int32_t>(frames[i].base);
            }
        }
        if (halted) {
            result[0] = signed_word(cells[0]);
            result[1] = static_cast<std::int32_t>(tick + 1);
            return ok;
        }
        pc = next;
    }
    if (trap) {
        trap[0] = static_cast<std::int64_t>(fuel);
        trap[1] = static_cast<std::int64_t>(pc);
    }
    return fuel_exhausted;
}
} // namespace

extern "C" void visuaml_structured_batch(
        const std::int32_t* arguments, const std::int32_t* fuels, std::size_t size,
        std::size_t count, std::size_t terminal, std::size_t visible_locals, std::size_t trace_stride,
        const double* rows, const double* fixed, const std::int32_t* targets, const double* guards,
        const double* guard_weights, const double* path_weights, const double*,
        const std::int64_t* flow, const double* event_defaults, const double* event_gains,
        const double* frame_gains, std::int32_t* results, std::int32_t* statuses,
        std::int32_t* trace_states, double* trace_routes, std::int32_t* trace_frames,
        std::uint8_t* memory, const std::uint8_t* memory_gains, std::uint32_t* trace_memory, std::int64_t* trap) {
    const auto state_stride = trace_stride * (2 + prepared_stack_size + visible_locals);
    for (std::size_t i = 0; i < size; ++i) {
        statuses[i] = structured_loop(
            arguments[i], fuels[i], count, terminal, visible_locals, rows, fixed,
            targets, guards, guard_weights, path_weights, flow, event_defaults, event_gains, frame_gains,
            results + i * 2, trace_states ? trace_states + i * state_stride : nullptr,
            trace_routes ? trace_routes + i * trace_stride * 5 : nullptr,
            trace_frames ? trace_frames + i * trace_stride * frame_fields : nullptr,
            memory, memory_gains, trace_memory, trap);
    }
}
