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

#если !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#если defined(_WIN32)
#    если defined(SPINE_TCM_BUILD_SHARED)
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
    целое    is_fake_tcm;
} spine_tcm_mem_info_t;

typedef struct spine_tcm_block_info {
    целое      id;
    void *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    целое      owner_tid;
    целое      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API const char * spine_tcm_runtime_version(void);
SPINE_TCM_API целое          spine_tcm_runtime_is_available(void);
SPINE_TCM_API целое          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API целое          spine_tcm_runtime_mem_info(целое id, spine_tcm_block_info_t * info);
SPINE_TCM_API void *       spine_tcm_runtime_mem_get(целое id);
SPINE_TCM_API целое          spine_tcm_runtime_mem_free(целое id);
SPINE_TCM_API void *       spine_tcm_runtime_mem_try_wait(целое id, size_t timeout_us);
SPINE_TCM_API целое          spine_tcm_runtime_mem_release(целое id);
SPINE_TCM_API целое          spine_tcm_runtime_mem_force_release(целое id);
SPINE_TCM_API целое          spine_tcm_runtime_mem_query(целое id);

#если defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
static inline целое spine_tcm_open_handle(const char * so_path) {
    (void) so_path;
    вернуть 0;
}

static inline const char * spine_tcm_version(void) {
    вернуть spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
static inline целое spine_tcm_is_available(void) {
    вернуть spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
static inline целое spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    вернуть spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
static inline целое spine_tcm_block_info(целое id, spine_tcm_block_info_t * info) {
    вернуть spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static inline void * spine_tcm_mem_get(целое id) {
    вернуть spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static inline целое spine_tcm_mem_free(целое id) {
    вернуть spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
static inline void * spine_tcm_mem_try_wait(целое id, size_t over_time) {
    вернуть spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static inline целое spine_tcm_mem_release(целое id) {
    вернуть spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
static inline целое spine_tcm_mem_force_release(целое id) {
    вернуть spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
static inline целое spine_tcm_mem_query(целое id) {
    вернуть spine_tcm_runtime_mem_query(id);
}
#elif !defined(SPINE_TCM_BUILD_SHARED)
typedef struct spine_tcm_handle {
    void * module_handle;
    целое    use_global_scope;
    целое    owns_module_handle;
    const char * (*runtime_version)(void);
    целое (*runtime_is_available)(void);
    целое (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    целое (*runtime_mem_info)(целое id, spine_tcm_block_info_t * info);
    void * (*runtime_mem_get)(целое id);
    целое (*runtime_mem_free)(целое id);
    void * (*runtime_mem_try_wait)(целое id, size_t over_time);
    целое (*runtime_mem_release)(целое id);
    целое (*runtime_mem_force_release)(целое id);
    целое (*runtime_mem_query)(целое id);
} spine_tcm_handle_t;

static inline spine_tcm_handle_t * spine_tcm_default_handle(void) {
    static spine_tcm_handle_t handle = { 0 };
    вернуть &handle;
}

static inline void spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    если (handle != NULL) {
        memset(handle, 0, sizeof(*handle));
    }
}

static inline целое spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    void * symbol_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (const char * (*) (void) ) dlsym(symbol_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (целое (*)(void)) dlsym(symbol_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (целое (*)(spine_tcm_mem_info_t *)) dlsym(symbol_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (целое (*)(целое, spine_tcm_block_info_t *)) dlsym(symbol_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (void * (*) (целое) ) dlsym(symbol_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (целое (*)(целое)) dlsym(symbol_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (void * (*) (целое, size_t)) dlsym(symbol_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (целое (*)(целое)) dlsym(symbol_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (целое (*)(целое)) dlsym(symbol_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (целое (*)(целое)) dlsym(symbol_scope, "spine_tcm_runtime_mem_query");

    вернуть handle->runtime_version != NULL && handle->runtime_is_available != NULL &&
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
static inline целое spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    если (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == NULL) {
        вернуть -1;
    }

    handle->use_global_scope = 1;
    вернуть spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
static inline целое spine_tcm_open_handle(const char * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    const char *         library  = (so_path != NULL && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    если (resolved->module_handle != NULL || resolved->use_global_scope) {
        вернуть 0;
    }

    если (spine_tcm_try_bind_global(resolved) == 0) {
        вернуть 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != NULL ? 1 : 0;

    если (resolved->module_handle == NULL) {
        spine_tcm_handle_reset(resolved);
        вернуть -1;
    }

    если (spine_tcm_handle_bind(resolved) != 0) {
        если (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        вернуть -1;
    }

    вернуть 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
static inline целое spine_tcm_is_available(void) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_is_available == NULL) {
        вернуть 0;
    }

    вернуть resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
static inline целое spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_layout_info == NULL) {
        вернуть -1;
    }

    вернуть resolved->runtime_layout_info(info);
}

static inline const char * spine_tcm_version(void) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_version == NULL) {
        вернуть "unknown";
    }

    вернуть resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
static inline целое spine_tcm_block_info(целое id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_info == NULL) {
        вернуть -1;
    }

    вернуть resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static inline void * spine_tcm_mem_get(целое id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        вернуть NULL;
    }

    если (resolved->runtime_mem_get == NULL) {
        вернуть NULL;
    }

    вернуть resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static inline целое spine_tcm_mem_free(целое id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_free == NULL) {
        вернуть -1;
    }

    вернуть resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
static inline void * spine_tcm_mem_try_wait(целое id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        вернуть NULL;
    }

    если (resolved->runtime_mem_try_wait == NULL) {
        вернуть NULL;
    }

    вернуть resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static inline целое spine_tcm_mem_release(целое id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_release == NULL) {
        вернуть -1;
    }

    вернуть resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
static inline целое spine_tcm_mem_force_release(целое id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == NULL) {
        вернуть -1;
    }

    вернуть resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
static inline целое spine_tcm_mem_query(целое id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    если (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    если ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_query == NULL) {
        вернуть -1;
    }

    вернуть resolved->runtime_mem_query(id);
}
#иначе
static inline const char * spine_tcm_version(void) {
    вернуть spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
