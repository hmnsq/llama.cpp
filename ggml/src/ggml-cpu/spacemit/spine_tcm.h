#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ SPINE_TCM_ਜਨਤਕ_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SPINE_TCM_ਜਨਤਕ_H

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

#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਜੇ !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_ਸਿੱਧਾ_LINK)
#    include <dlfcn.h>
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

#ਜੇ defined(_WIN32)
#    ਜੇ defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    ਨਹੀਂਤਾਂ
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#ਨਹੀਂਤਾਂ
#    define SPINE_TCM_API __attribute__((visibility("default")))
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ spine_tcm_mem_info {
    size_t blk_ਮਾਪ;
    size_t blk_ਅੰਕ;
    ਪੂਰਨ    is_fake_tcm;
} spine_tcm_mem_info_t;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ spine_tcm_ਖੰਡ_info {
    ਪੂਰਨ      id;
    ਖਾਲੀ *   va;
    size_t   ਮਾਪ;
    uint64_t phys_addr;
    uint64_t cpu_affinity_ਨਕਾਬ;
    ਪੂਰਨ      owner_tid;
    ਪੂਰਨ      is_acquired;
} spine_tcm_ਖੰਡ_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ਅਚਲ ਅੱਖਰ * spine_tcm_runtime_ਸੰਸਕਰਣ(ਖਾਲੀ);
SPINE_TCM_API ਪੂਰਨ          spine_tcm_runtime_is_ਉਪਲਬਧ(ਖਾਲੀ);
SPINE_TCM_API ਪੂਰਨ          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API ਪੂਰਨ          spine_tcm_runtime_mem_info(ਪੂਰਨ id, spine_tcm_ਖੰਡ_info_t * info);
SPINE_TCM_API ਖਾਲੀ *       spine_tcm_runtime_mem_get(ਪੂਰਨ id);
SPINE_TCM_API ਪੂਰਨ          spine_tcm_runtime_mem_free(ਪੂਰਨ id);
SPINE_TCM_API ਖਾਲੀ *       spine_tcm_runtime_mem_try_wait(ਪੂਰਨ id, size_t timeout_us);
SPINE_TCM_API ਪੂਰਨ          spine_tcm_runtime_mem_release(ਪੂਰਨ id);
SPINE_TCM_API ਪੂਰਨ          spine_tcm_runtime_mem_force_release(ਪੂਰਨ id);
SPINE_TCM_API ਪੂਰਨ          spine_tcm_runtime_mem_query(ਪੂਰਨ id);

#ਜੇ defined(SPINE_TCM_ਸਿੱਧਾ_LINK)
/* Optional no-op in direct-link mode. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_open_handle(ਅਚਲ ਅੱਖਰ * so_ਰਸਤਾ) {
    (ਖਾਲੀ) so_ਰਸਤਾ;
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ਅੱਖਰ * spine_tcm_ਸੰਸਕਰਣ(ਖਾਲੀ) {
    ਵਾਪਸ spine_tcm_runtime_ਸੰਸਕਰਣ();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_is_ਉਪਲਬਧ(ਖਾਲੀ) {
    ਵਾਪਸ spine_tcm_runtime_is_ਉਪਲਬਧ();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    ਵਾਪਸ spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_ਖੰਡ_info(ਪੂਰਨ id, spine_tcm_ਖੰਡ_info_t * info) {
    ਵਾਪਸ spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * spine_tcm_mem_get(ਪੂਰਨ id) {
    ਵਾਪਸ spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_free(ਪੂਰਨ id) {
    ਵਾਪਸ spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * spine_tcm_mem_try_wait(ਪੂਰਨ id, size_t over_ਸਮਾਂ) {
    ਵਾਪਸ spine_tcm_runtime_mem_try_wait(id, over_ਸਮਾਂ);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_release(ਪੂਰਨ id) {
    ਵਾਪਸ spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_force_release(ਪੂਰਨ id) {
    ਵਾਪਸ spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_query(ਪੂਰਨ id) {
    ਵਾਪਸ spine_tcm_runtime_mem_query(id);
}
#ਜਾਂਜੇ !defined(SPINE_TCM_BUILD_SHARED)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ spine_tcm_handle {
    ਖਾਲੀ * module_handle;
    ਪੂਰਨ    use_global_scope;
    ਪੂਰਨ    owns_module_handle;
    ਅਚਲ ਅੱਖਰ * (*runtime_ਸੰਸਕਰਣ)(ਖਾਲੀ);
    ਪੂਰਨ (*runtime_is_ਉਪਲਬਧ)(ਖਾਲੀ);
    ਪੂਰਨ (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    ਪੂਰਨ (*runtime_mem_info)(ਪੂਰਨ id, spine_tcm_ਖੰਡ_info_t * info);
    ਖਾਲੀ * (*runtime_mem_get)(ਪੂਰਨ id);
    ਪੂਰਨ (*runtime_mem_free)(ਪੂਰਨ id);
    ਖਾਲੀ * (*runtime_mem_try_wait)(ਪੂਰਨ id, size_t over_ਸਮਾਂ);
    ਪੂਰਨ (*runtime_mem_release)(ਪੂਰਨ id);
    ਪੂਰਨ (*runtime_mem_force_release)(ਪੂਰਨ id);
    ਪੂਰਨ (*runtime_mem_query)(ਪੂਰਨ id);
} spine_tcm_handle_t;

ਸਥਿਰ ਸਤਰਅੰਦਰ spine_tcm_handle_t * spine_tcm_default_handle(ਖਾਲੀ) {
    ਸਥਿਰ spine_tcm_handle_t handle = { 0 };
    ਵਾਪਸ &handle;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    ਜੇ (handle != ਨਿਰਾ) {
        memset(handle, 0, ਆਕਾਰ(*handle));
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    ਖਾਲੀ * ਚਿੰਨ੍ਹ_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_ਸੰਸਕਰਣ      = (ਅਚਲ ਅੱਖਰ * (*) (ਖਾਲੀ) ) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_version");
    handle->runtime_is_ਉਪਲਬਧ = (ਪੂਰਨ (*)(ਖਾਲੀ)) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (ਪੂਰਨ (*)(spine_tcm_mem_info_t *)) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (ਪੂਰਨ (*)(ਪੂਰਨ, spine_tcm_ਖੰਡ_info_t *)) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (ਖਾਲੀ * (*) (ਪੂਰਨ) ) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (ਪੂਰਨ (*)(ਪੂਰਨ)) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (ਖਾਲੀ * (*) (ਪੂਰਨ, size_t)) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (ਪੂਰਨ (*)(ਪੂਰਨ)) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (ਪੂਰਨ (*)(ਪੂਰਨ)) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (ਪੂਰਨ (*)(ਪੂਰਨ)) dlsym(ਚਿੰਨ੍ਹ_scope, "spine_tcm_runtime_mem_query");

    ਵਾਪਸ handle->runtime_ਸੰਸਕਰਣ != ਨਿਰਾ && handle->runtime_is_ਉਪਲਬਧ != ਨਿਰਾ &&
                   handle->runtime_layout_info != ਨਿਰਾ && handle->runtime_mem_info != ਨਿਰਾ &&
                   handle->runtime_mem_get != ਨਿਰਾ && handle->runtime_mem_free != ਨਿਰਾ &&
                   handle->runtime_mem_try_wait != ਨਿਰਾ && handle->runtime_mem_release != ਨਿਰਾ &&
                   handle->runtime_mem_force_release != ਨਿਰਾ && handle->runtime_mem_query != ਨਿਰਾ ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    ਜੇ (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == ਨਿਰਾ) {
        ਵਾਪਸ -1;
    }

    handle->use_global_scope = 1;
    ਵਾਪਸ spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_open_handle(ਅਚਲ ਅੱਖਰ * so_ਰਸਤਾ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ਅਚਲ ਅੱਖਰ *         library  = (so_ਰਸਤਾ != ਨਿਰਾ && so_ਰਸਤਾ[0] != '\0') ? so_ਰਸਤਾ : "libspine_tcm.so";

    ਜੇ (resolved->module_handle != ਨਿਰਾ || resolved->use_global_scope) {
        ਵਾਪਸ 0;
    }

    ਜੇ (spine_tcm_try_bind_global(resolved) == 0) {
        ਵਾਪਸ 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != ਨਿਰਾ ? 1 : 0;

    ਜੇ (resolved->module_handle == ਨਿਰਾ) {
        spine_tcm_handle_reset(resolved);
        ਵਾਪਸ -1;
    }

    ਜੇ (spine_tcm_handle_bind(resolved) != 0) {
        ਜੇ (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        ਵਾਪਸ -1;
    }

    ਵਾਪਸ 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_is_ਉਪਲਬਧ(ਖਾਲੀ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ ((resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) || resolved->runtime_is_ਉਪਲਬਧ == ਨਿਰਾ) {
        ਵਾਪਸ 0;
    }

    ਵਾਪਸ resolved->runtime_is_ਉਪਲਬਧ();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ ((resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) || resolved->runtime_layout_info == ਨਿਰਾ) {
        ਵਾਪਸ -1;
    }

    ਵਾਪਸ resolved->runtime_layout_info(info);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ਅੱਖਰ * spine_tcm_ਸੰਸਕਰਣ(ਖਾਲੀ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ ((resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) || resolved->runtime_ਸੰਸਕਰਣ == ਨਿਰਾ) {
        ਵਾਪਸ "unknown";
    }

    ਵਾਪਸ resolved->runtime_ਸੰਸਕਰਣ();
}

/* Returns per-block runtime metadata for the given TCM id. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_ਖੰਡ_info(ਪੂਰਨ id, spine_tcm_ਖੰਡ_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ ((resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) || resolved->runtime_mem_info == ਨਿਰਾ) {
        ਵਾਪਸ -1;
    }

    ਵਾਪਸ resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * spine_tcm_mem_get(ਪੂਰਨ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (resolved->runtime_mem_get == ਨਿਰਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_free(ਪੂਰਨ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ ((resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) || resolved->runtime_mem_free == ਨਿਰਾ) {
        ਵਾਪਸ -1;
    }

    ਵਾਪਸ resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * spine_tcm_mem_try_wait(ਪੂਰਨ id, size_t over_ਸਮਾਂ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (resolved->runtime_mem_try_wait == ਨਿਰਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ resolved->runtime_mem_try_wait(id, over_ਸਮਾਂ);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_release(ਪੂਰਨ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ ((resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) || resolved->runtime_mem_release == ਨਿਰਾ) {
        ਵਾਪਸ -1;
    }

    ਵਾਪਸ resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_force_release(ਪੂਰਨ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ ((resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == ਨਿਰਾ) {
        ਵਾਪਸ -1;
    }

    ਵਾਪਸ resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ spine_tcm_mem_query(ਪੂਰਨ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ਜੇ (resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) {
        (ਖਾਲੀ) spine_tcm_open_handle(ਨਿਰਾ);
    }

    ਜੇ ((resolved->module_handle == ਨਿਰਾ && !resolved->use_global_scope) || resolved->runtime_mem_query == ਨਿਰਾ) {
        ਵਾਪਸ -1;
    }

    ਵਾਪਸ resolved->runtime_mem_query(id);
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ਅੱਖਰ * spine_tcm_ਸੰਸਕਰਣ(ਖਾਲੀ) {
    ਵਾਪਸ spine_tcm_runtime_ਸੰਸਕਰਣ();
}
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SPINE_TCM_ਸੰਸਕਰਣ (spine_tcm_ਸੰਸਕਰਣ())

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ
