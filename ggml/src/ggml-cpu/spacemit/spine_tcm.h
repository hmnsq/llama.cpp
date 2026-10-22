#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ SPINE_TCM_ಬಹಿರಂಗ_H
#ವ್ಯಾಖ್ಯಾನಿಸು SPINE_TCM_ಬಹಿರಂಗ_H

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

#ಆದರೆ !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_ನೇರ_LINK)
#    include <dlfcn.h>
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(_WIN32)
#    ಆದರೆ defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    ಇಲ್ಲದಿದ್ದರೆ
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#ಇಲ್ಲದಿದ್ದರೆ
#    define SPINE_TCM_API __attribute__((visibility("default")))
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ spine_tcm_mem_info {
    size_t blk_ಗಾತ್ರ;
    size_t blk_ಅಂಕೆ;
    ಪೂರ್ಣಾಂಕ    is_fake_tcm;
} spine_tcm_mem_info_t;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ spine_tcm_ಖಂಡ_info {
    ಪೂರ್ಣಾಂಕ      id;
    ಶೂನ್ಯ *   va;
    size_t   ಗಾತ್ರ_2;
    uint64_t phys_addr;
    uint64_t cpu_affinity_ಮುಖವಾಡ;
    ಪೂರ್ಣಾಂಕ      owner_tid;
    ಪೂರ್ಣಾಂಕ      is_acquired;
} spine_tcm_ಖಂಡ_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * spine_tcm_runtime_ಆವೃತ್ತಿ(ಶೂನ್ಯ);
SPINE_TCM_API ಪೂರ್ಣಾಂಕ          spine_tcm_runtime_is_ಲಭ್ಯ(ಶೂನ್ಯ);
SPINE_TCM_API ಪೂರ್ಣಾಂಕ          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API ಪೂರ್ಣಾಂಕ          spine_tcm_runtime_mem_info(ಪೂರ್ಣಾಂಕ id, spine_tcm_ಖಂಡ_info_t * info);
SPINE_TCM_API ಶೂನ್ಯ *       spine_tcm_runtime_mem_get(ಪೂರ್ಣಾಂಕ id);
SPINE_TCM_API ಪೂರ್ಣಾಂಕ          spine_tcm_runtime_mem_free(ಪೂರ್ಣಾಂಕ id);
SPINE_TCM_API ಶೂನ್ಯ *       spine_tcm_runtime_mem_try_wait(ಪೂರ್ಣಾಂಕ id, size_t timeout_us);
SPINE_TCM_API ಪೂರ್ಣಾಂಕ          spine_tcm_runtime_mem_release(ಪೂರ್ಣಾಂಕ id);
SPINE_TCM_API ಪೂರ್ಣಾಂಕ          spine_tcm_runtime_mem_force_release(ಪೂರ್ಣಾಂಕ id);
SPINE_TCM_API ಪೂರ್ಣಾಂಕ          spine_tcm_runtime_mem_query(ಪೂರ್ಣಾಂಕ id);

#ಆದರೆ defined(SPINE_TCM_ನೇರ_LINK)
/* Optional no-op in direct-link mode. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_open_handle(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * so_ಮಾರ್ಗ) {
    (ಶೂನ್ಯ) so_ಮಾರ್ಗ;
    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * spine_tcm_ಆವೃತ್ತಿ(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_ಆವೃತ್ತಿ();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_is_ಲಭ್ಯ(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_is_ಲಭ್ಯ();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_ಖಂಡ_info(ಪೂರ್ಣಾಂಕ id, spine_tcm_ಖಂಡ_info_t * info) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * spine_tcm_mem_get(ಪೂರ್ಣಾಂಕ id) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_free(ಪೂರ್ಣಾಂಕ id) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * spine_tcm_mem_try_wait(ಪೂರ್ಣಾಂಕ id, size_t over_ಸಮಯ) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_mem_try_wait(id, over_ಸಮಯ);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_release(ಪೂರ್ಣಾಂಕ id) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_force_release(ಪೂರ್ಣಾಂಕ id) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_query(ಪೂರ್ಣಾಂಕ id) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_mem_query(id);
}
#ಅಥವಾಆದರೆ !defined(SPINE_TCM_BUILD_SHARED)
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ spine_tcm_handle {
    ಶೂನ್ಯ * module_handle;
    ಪೂರ್ಣಾಂಕ    use_global_scope;
    ಪೂರ್ಣಾಂಕ    owns_module_handle;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * (*runtime_ಆವೃತ್ತಿ)(ಶೂನ್ಯ);
    ಪೂರ್ಣಾಂಕ (*runtime_is_ಲಭ್ಯ)(ಶೂನ್ಯ);
    ಪೂರ್ಣಾಂಕ (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    ಪೂರ್ಣಾಂಕ (*runtime_mem_info)(ಪೂರ್ಣಾಂಕ id, spine_tcm_ಖಂಡ_info_t * info);
    ಶೂನ್ಯ * (*runtime_mem_get)(ಪೂರ್ಣಾಂಕ id);
    ಪೂರ್ಣಾಂಕ (*runtime_mem_free)(ಪೂರ್ಣಾಂಕ id);
    ಶೂನ್ಯ * (*runtime_mem_try_wait)(ಪೂರ್ಣಾಂಕ id, size_t over_ಸಮಯ);
    ಪೂರ್ಣಾಂಕ (*runtime_mem_release)(ಪೂರ್ಣಾಂಕ id);
    ಪೂರ್ಣಾಂಕ (*runtime_mem_force_release)(ಪೂರ್ಣಾಂಕ id);
    ಪೂರ್ಣಾಂಕ (*runtime_mem_query)(ಪೂರ್ಣಾಂಕ id);
} spine_tcm_handle_t;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ spine_tcm_handle_t * spine_tcm_default_handle(ಶೂನ್ಯ) {
    ಸ್ಥಿರ spine_tcm_handle_t handle = { 0 };
    ಹಿಂತಿರುಗಿ &handle;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    ಆದರೆ (handle != ಖಾಲಿ) {
        memset(handle, 0, ಗಾತ್ರ(*handle));
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    ಶೂನ್ಯ * ಚಿಹ್ನೆ_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_ಆವೃತ್ತಿ      = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * (*) (ಶೂನ್ಯ) ) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_version");
    handle->runtime_is_ಲಭ್ಯ = (ಪೂರ್ಣಾಂಕ (*)(ಶೂನ್ಯ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (ಪೂರ್ಣಾಂಕ (*)(spine_tcm_mem_info_t *)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (ಪೂರ್ಣಾಂಕ (*)(ಪೂರ್ಣಾಂಕ, spine_tcm_ಖಂಡ_info_t *)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (ಶೂನ್ಯ * (*) (ಪೂರ್ಣಾಂಕ) ) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (ಪೂರ್ಣಾಂಕ (*)(ಪೂರ್ಣಾಂಕ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (ಶೂನ್ಯ * (*) (ಪೂರ್ಣಾಂಕ, size_t)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (ಪೂರ್ಣಾಂಕ (*)(ಪೂರ್ಣಾಂಕ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (ಪೂರ್ಣಾಂಕ (*)(ಪೂರ್ಣಾಂಕ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (ಪೂರ್ಣಾಂಕ (*)(ಪೂರ್ಣಾಂಕ)) dlsym(ಚಿಹ್ನೆ_scope, "spine_tcm_runtime_mem_query");

    ಹಿಂತಿರುಗಿ handle->runtime_ಆವೃತ್ತಿ != ಖಾಲಿ && handle->runtime_is_ಲಭ್ಯ != ಖಾಲಿ &&
                   handle->runtime_layout_info != ಖಾಲಿ && handle->runtime_mem_info != ಖಾಲಿ &&
                   handle->runtime_mem_get != ಖಾಲಿ && handle->runtime_mem_free != ಖಾಲಿ &&
                   handle->runtime_mem_try_wait != ಖಾಲಿ && handle->runtime_mem_release != ಖಾಲಿ &&
                   handle->runtime_mem_force_release != ಖಾಲಿ && handle->runtime_mem_query != ಖಾಲಿ ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    ಆದರೆ (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    handle->use_global_scope = 1;
    ಹಿಂತಿರುಗಿ spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_open_handle(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * so_ಮಾರ್ಗ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *         library  = (so_ಮಾರ್ಗ != ಖಾಲಿ && so_ಮಾರ್ಗ[0] != '\0') ? so_ಮಾರ್ಗ : "libspine_tcm.so";

    ಆದರೆ (resolved->module_handle != ಖಾಲಿ || resolved->use_global_scope) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (spine_tcm_try_bind_global(resolved) == 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != ಖಾಲಿ ? 1 : 0;

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ) {
        spine_tcm_handle_reset(resolved);
        ಹಿಂತಿರುಗಿ -1;
    }

    ಆದರೆ (spine_tcm_handle_bind(resolved) != 0) {
        ಆದರೆ (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        ಹಿಂತಿರುಗಿ -1;
    }

    ಹಿಂತಿರುಗಿ 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_is_ಲಭ್ಯ(ಶೂನ್ಯ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ ((resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) || resolved->runtime_is_ಲಭ್ಯ == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_is_ಲಭ್ಯ();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ ((resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) || resolved->runtime_layout_info == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_layout_info(info);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * spine_tcm_ಆವೃತ್ತಿ(ಶೂನ್ಯ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ ((resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) || resolved->runtime_ಆವೃತ್ತಿ == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ "unknown";
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_ಆವೃತ್ತಿ();
}

/* Returns per-block runtime metadata for the given TCM id. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_ಖಂಡ_info(ಪೂರ್ಣಾಂಕ id, spine_tcm_ಖಂಡ_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ ((resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) || resolved->runtime_mem_info == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * spine_tcm_mem_get(ಪೂರ್ಣಾಂಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (resolved->runtime_mem_get == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_free(ಪೂರ್ಣಾಂಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ ((resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) || resolved->runtime_mem_free == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * spine_tcm_mem_try_wait(ಪೂರ್ಣಾಂಕ id, size_t over_ಸಮಯ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (resolved->runtime_mem_try_wait == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_mem_try_wait(id, over_ಸಮಯ);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_release(ಪೂರ್ಣಾಂಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ ((resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) || resolved->runtime_mem_release == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_force_release(ಪೂರ್ಣಾಂಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ ((resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ spine_tcm_mem_query(ಪೂರ್ಣಾಂಕ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ಆದರೆ (resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) {
        (ಶೂನ್ಯ) spine_tcm_open_handle(ಖಾಲಿ);
    }

    ಆದರೆ ((resolved->module_handle == ಖಾಲಿ && !resolved->use_global_scope) || resolved->runtime_mem_query == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಹಿಂತಿರುಗಿ resolved->runtime_mem_query(id);
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * spine_tcm_ಆವೃತ್ತಿ(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ spine_tcm_runtime_ಆವೃತ್ತಿ();
}
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು SPINE_TCM_ಆವೃತ್ತಿ (spine_tcm_ಆವೃತ್ತಿ())

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ
