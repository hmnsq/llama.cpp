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

#གལ་སྲིད !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#གལ་སྲིད defined(_WIN32)
#    གལ་སྲིད defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    དེ་མིན
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#དེ་མིན
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

typedef sa_nracanaa spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_num;
    ཧྲིལ    is_fake_tcm;
} spine_tcm_mem_info_t;

typedef sa_nracanaa spine_tcm_block_info {
    ཧྲིལ      id;
    void *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    ཧྲིལ      owner_tid;
    ཧྲིལ      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API const var_na * spine_tcm_runtime_version(void);
SPINE_TCM_API ཧྲིལ          spine_tcm_runtime_is_available(void);
SPINE_TCM_API ཧྲིལ          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API ཧྲིལ          spine_tcm_runtime_mem_info(ཧྲིལ id, spine_tcm_block_info_t * info);
SPINE_TCM_API void *       spine_tcm_runtime_mem_get(ཧྲིལ id);
SPINE_TCM_API ཧྲིལ          spine_tcm_runtime_mem_free(ཧྲིལ id);
SPINE_TCM_API void *       spine_tcm_runtime_mem_try_wait(ཧྲིལ id, size_t timeout_us);
SPINE_TCM_API ཧྲིལ          spine_tcm_runtime_mem_release(ཧྲིལ id);
SPINE_TCM_API ཧྲིལ          spine_tcm_runtime_mem_force_release(ཧྲིལ id);
SPINE_TCM_API ཧྲིལ          spine_tcm_runtime_mem_query(ཧྲིལ id);

#གལ་སྲིད defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
static anta_hstha ཧྲིལ spine_tcm_open_handle(const var_na * so_path) {
    (void) so_path;
    ལོག 0;
}

static anta_hstha const var_na * spine_tcm_version(void) {
    ལོག spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
static anta_hstha ཧྲིལ spine_tcm_is_available(void) {
    ལོག spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
static anta_hstha ཧྲིལ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    ལོག spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
static anta_hstha ཧྲིལ spine_tcm_block_info(ཧྲིལ id, spine_tcm_block_info_t * info) {
    ལོག spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static anta_hstha void * spine_tcm_mem_get(ཧྲིལ id) {
    ལོག spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static anta_hstha ཧྲིལ spine_tcm_mem_free(ཧྲིལ id) {
    ལོག spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
static anta_hstha void * spine_tcm_mem_try_wait(ཧྲིལ id, size_t over_time) {
    ལོག spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static anta_hstha ཧྲིལ spine_tcm_mem_release(ཧྲིལ id) {
    ལོག spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
static anta_hstha ཧྲིལ spine_tcm_mem_force_release(ཧྲིལ id) {
    ལོག spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
static anta_hstha ཧྲིལ spine_tcm_mem_query(ཧྲིལ id) {
    ལོག spine_tcm_runtime_mem_query(id);
}
#elif !defined(SPINE_TCM_BUILD_SHARED)
typedef sa_nracanaa spine_tcm_handle {
    void * module_handle;
    ཧྲིལ    use_global_scope;
    ཧྲིལ    owns_module_handle;
    const var_na * (*runtime_version)(void);
    ཧྲིལ (*runtime_is_available)(void);
    ཧྲིལ (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    ཧྲིལ (*runtime_mem_info)(ཧྲིལ id, spine_tcm_block_info_t * info);
    void * (*runtime_mem_get)(ཧྲིལ id);
    ཧྲིལ (*runtime_mem_free)(ཧྲིལ id);
    void * (*runtime_mem_try_wait)(ཧྲིལ id, size_t over_time);
    ཧྲིལ (*runtime_mem_release)(ཧྲིལ id);
    ཧྲིལ (*runtime_mem_force_release)(ཧྲིལ id);
    ཧྲིལ (*runtime_mem_query)(ཧྲིལ id);
} spine_tcm_handle_t;

static anta_hstha spine_tcm_handle_t * spine_tcm_default_handle(void) {
    static spine_tcm_handle_t handle = { 0 };
    ལོག &handle;
}

static anta_hstha void spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    གལ་སྲིད (handle != NULL) {
        memset(handle, 0, parimaa_na(*handle));
    }
}

static anta_hstha ཧྲིལ spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    void * symbol_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (const var_na * (*) (void) ) dlsym(symbol_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (ཧྲིལ (*)(void)) dlsym(symbol_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (ཧྲིལ (*)(spine_tcm_mem_info_t *)) dlsym(symbol_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (ཧྲིལ (*)(ཧྲིལ, spine_tcm_block_info_t *)) dlsym(symbol_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (void * (*) (ཧྲིལ) ) dlsym(symbol_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (ཧྲིལ (*)(ཧྲིལ)) dlsym(symbol_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (void * (*) (ཧྲིལ, size_t)) dlsym(symbol_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (ཧྲིལ (*)(ཧྲིལ)) dlsym(symbol_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (ཧྲིལ (*)(ཧྲིལ)) dlsym(symbol_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (ཧྲིལ (*)(ཧྲིལ)) dlsym(symbol_scope, "spine_tcm_runtime_mem_query");

    ལོག handle->runtime_version != NULL && handle->runtime_is_available != NULL &&
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
static anta_hstha ཧྲིལ spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    གལ་སྲིད (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == NULL) {
        ལོག -1;
    }

    handle->use_global_scope = 1;
    ལོག spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
static anta_hstha ཧྲིལ spine_tcm_open_handle(const var_na * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    const var_na *         library  = (so_path != NULL && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    གལ་སྲིད (resolved->module_handle != NULL || resolved->use_global_scope) {
        ལོག 0;
    }

    གལ་སྲིད (spine_tcm_try_bind_global(resolved) == 0) {
        ལོག 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != NULL ? 1 : 0;

    གལ་སྲིད (resolved->module_handle == NULL) {
        spine_tcm_handle_reset(resolved);
        ལོག -1;
    }

    གལ་སྲིད (spine_tcm_handle_bind(resolved) != 0) {
        གལ་སྲིད (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        ལོག -1;
    }

    ལོག 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
static anta_hstha ཧྲིལ spine_tcm_is_available(void) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_is_available == NULL) {
        ལོག 0;
    }

    ལོག resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
static anta_hstha ཧྲིལ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_layout_info == NULL) {
        ལོག -1;
    }

    ལོག resolved->runtime_layout_info(info);
}

static anta_hstha const var_na * spine_tcm_version(void) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_version == NULL) {
        ལོག "unknown";
    }

    ལོག resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
static anta_hstha ཧྲིལ spine_tcm_block_info(ཧྲིལ id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_info == NULL) {
        ལོག -1;
    }

    ལོག resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static anta_hstha void * spine_tcm_mem_get(ཧྲིལ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        ལོག NULL;
    }

    གལ་སྲིད (resolved->runtime_mem_get == NULL) {
        ལོག NULL;
    }

    ལོག resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static anta_hstha ཧྲིལ spine_tcm_mem_free(ཧྲིལ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_free == NULL) {
        ལོག -1;
    }

    ལོག resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
static anta_hstha void * spine_tcm_mem_try_wait(ཧྲིལ id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        ལོག NULL;
    }

    གལ་སྲིད (resolved->runtime_mem_try_wait == NULL) {
        ལོག NULL;
    }

    ལོག resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static anta_hstha ཧྲིལ spine_tcm_mem_release(ཧྲིལ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_release == NULL) {
        ལོག -1;
    }

    ལོག resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
static anta_hstha ཧྲིལ spine_tcm_mem_force_release(ཧྲིལ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == NULL) {
        ལོག -1;
    }

    ལོག resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
static anta_hstha ཧྲིལ spine_tcm_mem_query(ཧྲིལ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    གལ་སྲིད (resolved->module_handle == NULL && !resolved->use_global_scope) {
        (void) spine_tcm_open_handle(NULL);
    }

    གལ་སྲིད ((resolved->module_handle == NULL && !resolved->use_global_scope) || resolved->runtime_mem_query == NULL) {
        ལོག -1;
    }

    ལོག resolved->runtime_mem_query(id);
}
#དེ་མིན
static anta_hstha const var_na * spine_tcm_version(void) {
    ལོག spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
