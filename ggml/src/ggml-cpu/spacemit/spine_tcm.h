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

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಯದಿ !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

#ಯದಿ defined(_WIN32)
#    ಯದಿ defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    ಅನ್ಯಥಾ
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#ಅನ್ಯಥಾ
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_ಅಂಕೆ;
    ಪೂರ್ಣಾಙ್ಕ    is_fake_tcm;
} spine_tcm_mem_info_t;

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ spine_tcm_block_info {
    ಪೂರ್ಣಾಙ್ಕ      id;
    ಶೂನ್ಯ *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    ಪೂರ್ಣಾಙ್ಕ      owner_tid;
    ಪೂರ್ಣಾಙ್ಕ      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ಧ್ರುವ ವರ್ಣ * spine_tcm_runtime_version(ಶೂನ್ಯ);
SPINE_TCM_API ಪೂರ್ಣಾಙ್ಕ          spine_tcm_runtime_is_available(ಶೂನ್ಯ);
SPINE_TCM_API ಪೂರ್ಣಾಙ್ಕ          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API ಪೂರ್ಣಾಙ್ಕ          spine_tcm_runtime_mem_info(ಪೂರ್ಣಾಙ್ಕ id, spine_tcm_block_info_t * info);
SPINE_TCM_API ಶೂನ್ಯ *       spine_tcm_runtime_mem_get(ಪೂರ್ಣಾಙ್ಕ id);
SPINE_TCM_API ಪೂರ್ಣಾಙ್ಕ          spine_tcm_runtime_mem_free(ಪೂರ್ಣಾಙ್ಕ id);
SPINE_TCM_API ಶೂನ್ಯ *       spine_tcm_runtime_mem_try_wait(ಪೂರ್ಣಾಙ್ಕ id, size_t timeout_us);
SPINE_TCM_API ಪೂರ್ಣಾಙ್ಕ          spine_tcm_runtime_mem_release(ಪೂರ್ಣಾಙ್ಕ id);
SPINE_TCM_API ಪೂರ್ಣಾಙ್ಕ          spine_tcm_runtime_mem_force_release(ಪೂರ್ಣಾಙ್ಕ id);
SPINE_TCM_API ಪೂರ್ಣಾಙ್ಕ          spine_tcm_runtime_mem_query(ಪೂರ್ಣಾಙ್ಕ id);

#ಯದಿ defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_open_handle(ಧ್ರುವ ವರ್ಣ * so_path) {
    (ಶೂನ್ಯ) so_path;
    ಪ್ರತ್ಯಾಗಮ 0;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ ವರ್ಣ * spine_tcm_version(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_is_available(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_block_info(ಪೂರ್ಣಾಙ್ಕ id, spine_tcm_block_info_t * info) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ * spine_tcm_mem_get(ಪೂರ್ಣಾಙ್ಕ id) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_free(ಪೂರ್ಣಾಙ್ಕ id) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ * spine_tcm_mem_try_wait(ಪೂರ್ಣಾಙ್ಕ id, size_t over_time) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_release(ಪೂರ್ಣಾಙ್ಕ id) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_force_release(ಪೂರ್ಣಾಙ್ಕ id) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_query(ಪೂರ್ಣಾಙ್ಕ id) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_mem_query(id);
}
#ನೋಚೇದ್ಯದಿ !defined(SPINE_TCM_BUILD_SHARED)
ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ spine_tcm_handle {
    ಶೂನ್ಯ * module_handle;
    ಪೂರ್ಣಾಙ್ಕ    use_global_scope;
    ಪೂರ್ಣಾಙ್ಕ    owns_module_handle;
    ಧ್ರುವ ವರ್ಣ * (*runtime_version)(ಶೂನ್ಯ);
    ಪೂರ್ಣಾಙ್ಕ (*runtime_is_available)(ಶೂನ್ಯ);
    ಪೂರ್ಣಾಙ್ಕ (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    ಪೂರ್ಣಾಙ್ಕ (*runtime_mem_info)(ಪೂರ್ಣಾಙ್ಕ id, spine_tcm_block_info_t * info);
    ಶೂನ್ಯ * (*runtime_mem_get)(ಪೂರ್ಣಾಙ್ಕ id);
    ಪೂರ್ಣಾಙ್ಕ (*runtime_mem_free)(ಪೂರ್ಣಾಙ್ಕ id);
    ಶೂನ್ಯ * (*runtime_mem_try_wait)(ಪೂರ್ಣಾಙ್ಕ id, size_t over_time);
    ಪೂರ್ಣಾಙ್ಕ (*runtime_mem_release)(ಪೂರ್ಣಾಙ್ಕ id);
    ಪೂರ್ಣಾಙ್ಕ (*runtime_mem_force_release)(ಪೂರ್ಣಾಙ್ಕ id);
    ಪೂರ್ಣಾಙ್ಕ (*runtime_mem_query)(ಪೂರ್ಣಾಙ್ಕ id);
} spine_tcm_handle_t;

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ spine_tcm_handle_t * spine_tcm_default_handle(ಶೂನ್ಯ) {
    ಸ್ಥಿರ spine_tcm_handle_t handle = { 0 };
    ಪ್ರತ್ಯಾಗಮ &handle;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    ಯದಿ (handle != ರಿಕ್ತ) {
        memset(handle, 0, ಪರಿಮಾಣ(*handle));
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    ಶೂನ್ಯ * ಚಿಹ್ನೆ_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (ಧ್ರುವ ವರ್ಣ * (*) (ಶೂನ್ಯ) ) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (ಪೂರ್ಣಾಙ್ಕ (*)(ಶೂನ್ಯ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (ಪೂರ್ಣಾಙ್ಕ (*)(spine_tcm_mem_info_t *)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (ಪೂರ್ಣಾಙ್ಕ (*)(ಪೂರ್ಣಾಙ್ಕ, spine_tcm_block_info_t *)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (ಶೂನ್ಯ * (*) (ಪೂರ್ಣಾಙ್ಕ) ) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (ಪೂರ್ಣಾಙ್ಕ (*)(ಪೂರ್ಣಾಙ್ಕ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (ಶೂನ್ಯ * (*) (ಪೂರ್ಣಾಙ್ಕ, size_t)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (ಪೂರ್ಣಾಙ್ಕ (*)(ಪೂರ್ಣಾಙ್ಕ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (ಪೂರ್ಣಾಙ್ಕ (*)(ಪೂರ್ಣಾಙ್ಕ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (ಪೂರ್ಣಾಙ್ಕ (*)(ಪೂರ್ಣಾಙ್ಕ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_query");

    ಪ್ರತ್ಯಾಗಮ handle->runtime_version != ರಿಕ್ತ && handle->runtime_is_available != ರಿಕ್ತ &&
                   handle->runtime_layout_info != ರಿಕ್ತ && handle->runtime_mem_info != ರಿಕ್ತ &&
                   handle->runtime_mem_get != ರಿಕ್ತ && handle->runtime_mem_free != ರಿಕ್ತ &&
                   handle->runtime_mem_try_wait != ರಿಕ್ತ && handle->runtime_mem_release != ರಿಕ್ತ &&
                   handle->runtime_mem_force_release != ರಿಕ್ತ && handle->runtime_mem_query != ರಿಕ್ತ ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    ಯದಿ (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    handle->use_global_scope = 1;
    ಪ್ರತ್ಯಾಗಮ spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_open_handle(ಧ್ರುವ ವರ್ಣ * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ಧ್ರುವ ವರ್ಣ *         library  = (so_path != ರಿಕ್ತ && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    ಯದಿ (resolved->module_handle != ರಿಕ್ತ || resolved->use_global_scope) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (spine_tcm_try_bind_global(resolved) == 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != ರಿಕ್ತ ? 1 : 0;

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ) {
        spine_tcm_handle_reset(resolved);
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಯದಿ (spine_tcm_handle_bind(resolved) != 0) {
        ಯದಿ (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_is_available(ಶೂನ್ಯ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ ((resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) || resolved->runtime_is_available == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ ((resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) || resolved->runtime_layout_info == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_layout_info(info);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ ವರ್ಣ * spine_tcm_version(ಶೂನ್ಯ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ ((resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) || resolved->runtime_version == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ "unknown";
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_block_info(ಪೂರ್ಣಾಙ್ಕ id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ ((resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) || resolved->runtime_mem_info == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ * spine_tcm_mem_get(ಪೂರ್ಣಾಙ್ಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (resolved->runtime_mem_get == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_free(ಪೂರ್ಣಾಙ್ಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ ((resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) || resolved->runtime_mem_free == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ * spine_tcm_mem_try_wait(ಪೂರ್ಣಾಙ್ಕ id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (resolved->runtime_mem_try_wait == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_release(ಪೂರ್ಣಾಙ್ಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ ((resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) || resolved->runtime_mem_release == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_force_release(ಪೂರ್ಣಾಙ್ಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ ((resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ spine_tcm_mem_query(ಪೂರ್ಣಾಙ್ಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಯದಿ (resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ರಿಕ್ತ);
    }

    ಯದಿ ((resolved->module_handle == ರಿಕ್ತ && !resolved->use_global_scope) || resolved->runtime_mem_query == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪ್ರತ್ಯಾಗಮ resolved->runtime_mem_query(id);
}
#ಅನ್ಯಥಾ
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ ವರ್ಣ * spine_tcm_version(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
