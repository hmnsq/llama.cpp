#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#അന്തര്ഭാവ <stddef.h>
#അന്തര്ഭാവ <stdint.h>

സ്ഥിതം വരിയിൽ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    തിരികെ p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) ചെയ്യുക { (L)->field = (off); (off) += (sz); } ആകുമ്പോൾ (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) ചെയ്യുക { എങ്കിൽ (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } അല്ലെങ്കിൽ { (L)->field = 0; } } ആകുമ്പോൾ (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : ശൂന്യ)

#endif // HTP_VTCM_H
