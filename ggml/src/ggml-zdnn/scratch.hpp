#నిర్వచితం_కాకపోతే GGML_ZDNN_SCRATCH_HPP
#నిర్వచించు GGML_ZDNN_SCRATCH_HPP

#అన్తర్భావ "ggml.h"

#అన్తర్భావ <cstddef>
#అన్తర్భావ <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
నిర్మాణం ggml_zdnn_pool {
    నిర్మాణం బఫరు {
        శూన్య * ptr  = శూన్యసూచిక;
        size_t size = 0;
    };

    std::vector<బఫరు> cached;

    ~ggml_zdnn_pool();

    శూన్య * alloc(size_t size, size_t * actual_కొలత);
    శూన్య   విడుదల(శూన్య * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
మూస <రకం_పేరు T>
నిర్మాణం ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = శూన్యసూచిక;
    T * ptr = శూన్యసూచిక;
    size_t actual_కొలత = 0;

    ggml_zdnn_pool_alloc() = అప్రమేయ;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        ఒకవేళ (ptr != శూన్యసూచిక) {
            pool->విడుదల(ptr, actual_కొలత);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == శూన్యసూచిక);
        ఇది->pool = &pool;
        ptr = (T *) pool.alloc(size * పరిమాణం(T), &actual_కొలత);
        తిరిగి ptr;
    }

    T * get() {
        తిరిగి ptr;
    }

    ggml_zdnn_pool_alloc(స్థిరరాశి ggml_zdnn_pool_alloc &) = తొలగించు;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = తొలగించు;
    ggml_zdnn_pool_alloc & క్రియాకారకం=(స్థిరరాశి ggml_zdnn_pool_alloc &) = తొలగించు;
    ggml_zdnn_pool_alloc & క్రియాకారకం=(ggml_zdnn_pool_alloc &&) = తొలగించు;
};

#అయితే_ముగింపు  // GGML_ZDNN_SCRATCH_HPP
