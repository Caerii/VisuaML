#include <stddef.h>
#include <stdint.h>

/* Opcodes follow the reviewed v60 affine fragment. Arithmetic wraps in u32. */
int affine_v61(const int32_t *code, size_t count, uint32_t *result) {
    uint32_t a[512], b[512];
    size_t depth = 0;
    if (count == 0 || count > 512)
        return 0;
    for (size_t i = 0; i < count; ++i) {
        int32_t op = code[2*i], imm = code[2*i+1];
        if (op == 0 || op == 1) {
            if (op == 0 && imm != 0)
                return 0;
            a[depth] = op == 0;
            b[depth++] = op == 0 ? 0 : (uint32_t)imm;
            continue;
        }
        if (depth < 2 || imm != 0 || op < 2 || op > 4)
            return 0;
        --depth;
        uint32_t c = a[depth], d = b[depth];
        uint32_t *x = &a[depth-1], *y = &b[depth-1];
        if (op == 4) {
            if (*x != 0 && c != 0)
                return 0;
            *x = *x*d + *y*c;
            *y *= d;
        } else if (op == 2) {
            *x += c;
            *y += d;
        } else {
            *x -= c;
            *y -= d;
        }
    }
    if (depth != 1)
        return 0;
    result[0] = a[0];
    result[1] = b[0];
    return 1;
}
