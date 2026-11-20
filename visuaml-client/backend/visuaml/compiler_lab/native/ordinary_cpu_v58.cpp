// Bounded CPU lowering of the prepared non-memory transition.
// The input image is validated WASM data. Memory, cases with region
// metadata, and call/frame boundaries are handed back to the graph.
#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <tuple>

namespace {
constexpr int64_t kStack = 24;
constexpr int64_t kLocals = 5;
constexpr int64_t kMask = 0xffffffff;

int64_t signed_word(int64_t word) {
    return (word ^ 0x80000000) - 0x80000000;
}

using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                          at::Tensor, at::Tensor, int64_t>;

Result ordinary_run(const at::Tensor& image, const at::Tensor& pc_tensor,
                    const at::Tensor& height_tensor, const at::Tensor& old_stack,
                    const at::Tensor& old_locals, const at::Tensor& fuel_tensor,
                    int64_t budget) {
    TORCH_CHECK(image.dim() == 2 && image.size(1) == 18 && image.is_contiguous(),
                "Expected contiguous prepared ordinary image");
    TORCH_CHECK(old_stack.dim() == 1 && old_stack.numel() == kStack &&
                old_stack.is_contiguous() && old_locals.dim() == 1 &&
                old_locals.numel() == kLocals && old_locals.is_contiguous(),
                "Unexpected ordinary stack or locals");
    TORCH_CHECK(budget >= 0 && budget <= 64, "Unexpected transition budget");
    const auto* rows = image.const_data_ptr<int64_t>();
    const auto terminal = image.size(0) - 1;
    int64_t pc = pc_tensor.item<int64_t>();
    int64_t height = height_tensor.item<int64_t>();
    int64_t fuel = fuel_tensor.item<int64_t>();
    TORCH_CHECK(pc >= 0 && pc <= terminal && height >= 0 && height <= kStack && fuel >= 0,
                "Invalid ordinary entry state");
    std::array<int64_t, kStack> stack;
    std::array<int64_t, kLocals> locals;
    std::memcpy(stack.data(), old_stack.const_data_ptr<int64_t>(), sizeof(stack));
    std::memcpy(locals.data(), old_locals.const_data_ptr<int64_t>(), sizeof(locals));
    bool stack_dirty = false;
    bool locals_dirty = false;
    int64_t steps = 0;
    while (steps < budget && fuel > 0 && pc != terminal) {
        const auto* row = rows + pc * 18;
        const int64_t op = row[0];
        if (row[17] != 1 || op < 0 || op >= 20 || row[11] >= 0 || row[16] != 0) {
            break;
        }
        const int64_t pops = row[14];
        const int64_t pushes = row[15];
        TORCH_CHECK(pops >= 0 && pops <= 2 && pushes >= 0 && pushes <= 1 &&
                    height >= pops && (op != 12 || height >= 1),
                    "Invalid ordinary stack effect");
        const auto base = height - pops;
        bool take = row[5] != 0;
        if (op == 3 || op == 7) {
            take = op == 3 ? stack[height - 1] == 0 : stack[height - 1] != 0;
        }
        const auto next_pc = row[take ? 4 : 3];
        const auto next_height = (op == 6 || op == 7) && take ? row[8] : base + pushes;
        TORCH_CHECK(next_pc >= 0 && next_pc <= terminal && next_height >= 0 &&
                    next_height <= kStack && (!pushes || base < kStack),
                    "Invalid ordinary successor");
        if (op == 10 || op == 11 || op == 12) {
            TORCH_CHECK(row[1] >= 0 && row[1] < kLocals, "Invalid local slot");
        }
        if (pushes != 0) {
            const auto right = pops > 0 ? stack[height - 1] : 0;
            const auto left = pops == 2 ? stack[height - 2] : 0;
            int64_t value = 0;
            switch (op) {
                case 10: value = locals[row[1]]; break;
                case 13: value = row[1]; break;
                case 14: value = right == 0; break;
                case 15: value = signed_word(left) < signed_word(right); break;
                case 16: value = signed_word(left) > signed_word(right); break;
                case 17: value = (left + right) & kMask; break;
                case 18: value = (left - right) & kMask; break;
                case 19: value = (signed_word(left) * signed_word(right)) & kMask; break;
                default: TORCH_CHECK(false, "Invalid ordinary push opcode");
            }
            stack[base] = value;
            stack_dirty = true;
        }
        if (op == 11 || op == 12) {
            locals[row[1]] = stack[height - 1];
            locals_dirty = true;
        }
        if (next_height < height) {
            std::fill(stack.begin() + next_height, stack.begin() + height, 0);
            stack_dirty = true;
        }
        pc = next_pc;
        height = next_height;
        --fuel;
        ++steps;
    }

    auto stack_tensor = old_stack;
    auto locals_tensor = old_locals;
    if (stack_dirty) {
        stack_tensor = at::empty_like(old_stack);
        std::memcpy(stack_tensor.data_ptr<int64_t>(), stack.data(), sizeof(stack));
    }
    if (locals_dirty) {
        locals_tensor = at::empty_like(old_locals);
        std::memcpy(locals_tensor.data_ptr<int64_t>(), locals.data(), sizeof(locals));
    }
    return {at::scalar_tensor(pc, pc_tensor.options()),
            at::scalar_tensor(height, height_tensor.options()), stack_tensor, locals_tensor,
            at::scalar_tensor(fuel, fuel_tensor.options()),
            at::scalar_tensor(pc == terminal ? int64_t{1} : int64_t{0}, pc_tensor.options()), steps};
}
}  // namespace

TORCH_LIBRARY(visuaml_ordinary, m) {
    m.def("run(Tensor image, Tensor pc, Tensor height, Tensor(a) stack, Tensor(b) locals, "
          "Tensor fuel, int budget) -> (Tensor, Tensor, Tensor(a), Tensor(b), Tensor, Tensor, int)");
}
TORCH_LIBRARY_IMPL(visuaml_ordinary, CPU, m) {
    m.impl("run", ordinary_run);
}
