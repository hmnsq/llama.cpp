#अगर_अघोषित HTP_DCACHE_H
#घोषणा HTP_DCACHE_H

#समावेश <stdint.h>
#समावेश "htp-ops.h"

#घोषणा HTP_MAX_DIRTY_RANGES 64

काष्ठा htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

काष्ठा htp_context;

व्योम htp_dcache_flush_dirty_ranges(काष्ठा htp_context * ctx);
व्योम htp_dcache_flush_dirty_tensors(काष्ठा htp_context * ctx, स्थिर काष्ठा htp_tensor * स्थिर * tensors, uint32_t n);
व्योम htp_dcache_mark_dirty_tensors(काष्ठा htp_context * ctx, स्थिर काष्ठा htp_tensor * स्थिर * tensors, uint32_t n);

#पूर्ण_अगर // HTP_DCACHE_H
