#ifndef SPINE_TCM_PUBLIC_H_
#define SPINE_TCM_PUBLIC_H_

/*
 * spine_tcm public API
 *
 * Usage:
 *   1. Direct link mode
 *      Define SPINE_TCM_DIRECT_LINK and link against libspine_tcm.so.
 *
 *      if (spine_tcm_is_available()) {
 *          void *buffer = spine_tcm_mem_get(0);
 *          spine_tcm_mem_free(0);
 *      }
 *
 *   2. Header-only loader mode
 *      Include this header without linking libspine_tcm.so. The loader first
 *      tries to reuse a process-global spine_tcm instance and falls back to
 *      dlopen("libspine_tcm.so") when needed.
 *
 *      spine_tcm_open_handle(NULL);  // optional pre-bind
 *      if (spine_tcm_is_available()) {
 *          void *buffer = spine_tcm_mem_get(0);
 *          spine_tcm_mem_free(0);
 *      }
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ако !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ако defined(_WIN32)
#    ако defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    иначе
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#иначе
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

typedef struct spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_num;
    цяло    is_fake_tcm;
} spine_tcm_mem_info_t;

typedef struct spine_tcm_block_info {
    цяло      id;
    void *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    цяло      owner_tid;
    цяло      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API const char * spine_tcm_runtime_version(void);
SPINE_TCM_API цяло          spine_tcm_runtime_is_available(void);
SPINE_TCM_API цяло          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API цяло          spine_tcm_runtime_mem_info(цяло id, spine_tcm_block_info_t * info);
SPINE_TCM_API void *       spine_tcm_runtime_mem_get(цяло id);
SPINE_TCM_API цяло          spine_tcm_runtime_mem_free(цяло id);
SPINE_TCM_API void *       spine_tcm_runtime_mem_try_wait(цяло id, size_t timeout_us);
SPINE_TCM_API цяло          spine_tcm_runtime_mem_release(цяло id);
SPINE_TCM_API цяло          spine_tcm_runtime_mem_force_release(цяло id);
SPINE_TCM_API цяло          spine_tcm_runtime_mem_query(цяло id);

#ако defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
static inline цяло spine_tcm_open_handle(const char * so_path) {
    (void) so_path;
    върни 0;
}

static inline const char * spine_tcm_version(void) {
    върни spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
static inline цяло spine_tcm_is_available(void) {
    върни spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
static inline цяло spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    върни spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
static inline цяло spine_tcm_block_info(цяло id, spine_tcm_block_info_t * info) {
    върни spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static inline void * spine_tcm_mem_get(цяло id) {
    върни spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static inline цяло spine_tcm_mem_free(цяло id) {
    върни spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
static inline void * spine_tcm_mem_try_wait(цяло id, size_t over_time) {
    върни spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static inline цяло spine_tcm_mem_release(цяло id) {
    върни spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
static inline цяло spine_tcm_mem_force_release(цяло id) {
    върни spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
static inline цяло spine_tcm_mem_query(цяло id) {
    върни spine_tcm_runtime_mem_query(id);
}
#elif !defined(SPINE_TCM_BUILD_SHARED)
typedef struct spine_tcm_handle {
    void * module_handle;
    цяло    use_global_scope;
    цяло    owns_module_handle;
    const char * (*runtime_version)(void);
    цяло (*runtime_is_available)(void);
    цяло (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    цяло (*runtime_mem_info)(цяло id, spine_tcm_block_info_t * info);
    void * (*runtime_mem_get)(цяло id);
    цяло (*runtime_mem_free)(цяло id);
    void * (*runtime_mem_try_wait)(цяло id, size_t over_time);
    цяло (*runtime_mem_release)(цяло id);
    цяло (*runtime_mem_force_release)(цяло id);
    цяло (*runtime_mem_query)(цяло id);
} spine_tcm_handle_t;

static inline spine_tcm_handle_t * spine_tcm_default_handle(void) {
    static spine_tcm_handle_t handle = { 0 };
    върни &handle;
}

static inline void spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    ако (handle != NULL) {
        memset(handle, 0, sizeof(*handle));
    }
}

static inline цяло spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    void * symbol_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (const char * (*) (void) ) dlsym(symbol_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (цяло (*)(void)) dlsym(symbol_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (цяло (*)(spine_tcm_mem_info_t *)) dlsym(symbol_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (цяло (*)(цяло, spine_tcm_block_info_t *)) dlsym(symbol_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (void * (*) (цяло) ) dlsym(symbol_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (цяло (*)(цяло)) dlsym(symbol_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (void * (*) (цяло, size_t)) dlsym(symbol_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (цяло (*)(цяло)) dlsym(symbol_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (цяло (*)(цяло)) dlsym(symbol_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (цяло (*)(цяло)) dlsym(symbol_scope, "spine_tcm_runtime_mem_query");

    върни handle->runtime_version != NULL && handle->runtime_is_available != NULL &&
                   handle->runtime_layout_info != NULL && handle->runtime_mem_info != NULL &&
                   handle->runtime_mem_get != NULL && handle->runtime_mem_free != NULL &&
                   handle->runtime_mem_try_wait != NULL && handle->runtime_mem_release != NULL &&
                   handle->runtime_mem_force_release != NULL && handle->runtime_mem_query != NULL ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
static inline цяло spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    ако (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == NULL) {
        върни -1;
    }

    handle->use_global_scope = 1;
    върни spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
static inline цяло spine_tcm_open_handle(const char * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    const char *         library  = (so_path != NULL && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    ако (resolved->module_handle != NULL || resolved->use_global_scope) {
        върни 0;
    }

    ако (spine_tcm_try_bind_global(resolved) == 0) {
        върни 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != NULL ? 1 : 0;

    ако (resolved->module_handle == NULL) {
        spine_tcm_handle_reset(resolved);
        върни -1;
    }

    ако (spine_tcm_handle_bind(resolved) != 0) {
        ако (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        върни -1;
    }

    върни 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
static inline цяло spine_tcm_is_available(void) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_is_available == NULL) {
        върни 0;
    }

    върни resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
static inline цяло spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_layout_info == NULL) {
        върни -1;
    }

    върни resolved->runtime_layout_info(info);
}

static inline const char * spine_tcm_version(void) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_version == NULL) {
        върни "unknown";
    }

    върни resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
static inline цяло spine_tcm_block_info(цяло id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_info == NULL) {
        върни -1;
    }

    върни resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static inline void * spine_tcm_mem_get(цяло id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        върни NULL;
    }

    ако (resolved->runtime_mem_get == NULL) {
        върни NULL;
    }

    върни resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static inline цяло spine_tcm_mem_free(цяло id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_free == NULL) {
        върни -1;
    }

    върни resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
static inline void * spine_tcm_mem_try_wait(цяло id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        върни NULL;
    }

    ако (resolved->runtime_mem_try_wait == NULL) {
        върни NULL;
    }

    върни resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static inline цяло spine_tcm_mem_release(цяло id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_release == NULL) {
        върни -1;
    }

    върни resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
static inline цяло spine_tcm_mem_force_release(цяло id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == NULL) {
        върни -1;
    }

    върни resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
static inline цяло spine_tcm_mem_query(цяло id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ако (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    ако ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_query == NULL) {
        върни -1;
    }

    върни resolved->runtime_mem_query(id);
}
#иначе
static inline const char * spine_tcm_version(void) {
    върни spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
