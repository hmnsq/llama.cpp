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

#അന്തര്ഭാവ <stddef.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <string.h>

#എങ്കിൽ !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

#എങ്കിൽ defined(_WIN32)
#    എങ്കിൽ defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    അല്ലെങ്കിൽ
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#അല്ലെങ്കിൽ
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

തരനിർവചനം ഘടന spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_അക്കം;
    പൂർണ്ണസംഖ്യ    is_fake_tcm;
} spine_tcm_mem_info_t;

തരനിർവചനം ഘടന spine_tcm_block_info {
    പൂർണ്ണസംഖ്യ      id;
    ശൂന്യം *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    പൂർണ്ണസംഖ്യ      owner_tid;
    പൂർണ്ണസംഖ്യ      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API സ്ഥിരം അക്ഷരം * spine_tcm_runtime_version(ശൂന്യം);
SPINE_TCM_API പൂർണ്ണസംഖ്യ          spine_tcm_runtime_is_available(ശൂന്യം);
SPINE_TCM_API പൂർണ്ണസംഖ്യ          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API പൂർണ്ണസംഖ്യ          spine_tcm_runtime_mem_info(പൂർണ്ണസംഖ്യ id, spine_tcm_block_info_t * info);
SPINE_TCM_API ശൂന്യം *       spine_tcm_runtime_mem_get(പൂർണ്ണസംഖ്യ id);
SPINE_TCM_API പൂർണ്ണസംഖ്യ          spine_tcm_runtime_mem_free(പൂർണ്ണസംഖ്യ id);
SPINE_TCM_API ശൂന്യം *       spine_tcm_runtime_mem_try_wait(പൂർണ്ണസംഖ്യ id, size_t timeout_us);
SPINE_TCM_API പൂർണ്ണസംഖ്യ          spine_tcm_runtime_mem_release(പൂർണ്ണസംഖ്യ id);
SPINE_TCM_API പൂർണ്ണസംഖ്യ          spine_tcm_runtime_mem_force_release(പൂർണ്ണസംഖ്യ id);
SPINE_TCM_API പൂർണ്ണസംഖ്യ          spine_tcm_runtime_mem_query(പൂർണ്ണസംഖ്യ id);

#എങ്കിൽ defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_open_handle(സ്ഥിരം അക്ഷരം * so_path) {
    (ശൂന്യം) so_path;
    തിരികെ 0;
}

സ്ഥിതം വരിയിൽ സ്ഥിരം അക്ഷരം * spine_tcm_version(ശൂന്യം) {
    തിരികെ spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_is_available(ശൂന്യം) {
    തിരികെ spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    തിരികെ spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_block_info(പൂർണ്ണസംഖ്യ id, spine_tcm_block_info_t * info) {
    തിരികെ spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
സ്ഥിതം വരിയിൽ ശൂന്യം * spine_tcm_mem_get(പൂർണ്ണസംഖ്യ id) {
    തിരികെ spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_free(പൂർണ്ണസംഖ്യ id) {
    തിരികെ spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
സ്ഥിതം വരിയിൽ ശൂന്യം * spine_tcm_mem_try_wait(പൂർണ്ണസംഖ്യ id, size_t over_time) {
    തിരികെ spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_release(പൂർണ്ണസംഖ്യ id) {
    തിരികെ spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_force_release(പൂർണ്ണസംഖ്യ id) {
    തിരികെ spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_query(പൂർണ്ണസംഖ്യ id) {
    തിരികെ spine_tcm_runtime_mem_query(id);
}
#അല്ലെങ്കിലെങ്കിൽ !defined(SPINE_TCM_BUILD_SHARED)
തരനിർവചനം ഘടന spine_tcm_handle {
    ശൂന്യം * module_handle;
    പൂർണ്ണസംഖ്യ    use_global_scope;
    പൂർണ്ണസംഖ്യ    owns_module_handle;
    സ്ഥിരം അക്ഷരം * (*runtime_version)(ശൂന്യം);
    പൂർണ്ണസംഖ്യ (*runtime_is_available)(ശൂന്യം);
    പൂർണ്ണസംഖ്യ (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    പൂർണ്ണസംഖ്യ (*runtime_mem_info)(പൂർണ്ണസംഖ്യ id, spine_tcm_block_info_t * info);
    ശൂന്യം * (*runtime_mem_get)(പൂർണ്ണസംഖ്യ id);
    പൂർണ്ണസംഖ്യ (*runtime_mem_free)(പൂർണ്ണസംഖ്യ id);
    ശൂന്യം * (*runtime_mem_try_wait)(പൂർണ്ണസംഖ്യ id, size_t over_time);
    പൂർണ്ണസംഖ്യ (*runtime_mem_release)(പൂർണ്ണസംഖ്യ id);
    പൂർണ്ണസംഖ്യ (*runtime_mem_force_release)(പൂർണ്ണസംഖ്യ id);
    പൂർണ്ണസംഖ്യ (*runtime_mem_query)(പൂർണ്ണസംഖ്യ id);
} spine_tcm_handle_t;

സ്ഥിതം വരിയിൽ spine_tcm_handle_t * spine_tcm_default_handle(ശൂന്യം) {
    സ്ഥിതം spine_tcm_handle_t handle = { 0 };
    തിരികെ &handle;
}

സ്ഥിതം വരിയിൽ ശൂന്യം spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    എങ്കിൽ (handle != ശൂന്യ) {
        memset(handle, 0, വലുപ്പം(*handle));
    }
}

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    ശൂന്യം * ചിഹ്നം_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (സ്ഥിരം അക്ഷരം * (*) (ശൂന്യം) ) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (പൂർണ്ണസംഖ്യ (*)(ശൂന്യം)) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (പൂർണ്ണസംഖ്യ (*)(spine_tcm_mem_info_t *)) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (പൂർണ്ണസംഖ്യ (*)(പൂർണ്ണസംഖ്യ, spine_tcm_block_info_t *)) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (ശൂന്യം * (*) (പൂർണ്ണസംഖ്യ) ) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (പൂർണ്ണസംഖ്യ (*)(പൂർണ്ണസംഖ്യ)) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (ശൂന്യം * (*) (പൂർണ്ണസംഖ്യ, size_t)) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (പൂർണ്ണസംഖ്യ (*)(പൂർണ്ണസംഖ്യ)) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (പൂർണ്ണസംഖ്യ (*)(പൂർണ്ണസംഖ്യ)) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (പൂർണ്ണസംഖ്യ (*)(പൂർണ്ണസംഖ്യ)) dlsym(ചിഹ്നം_scope, "spine_tcm_runtime_mem_query");

    തിരികെ handle->runtime_version != ശൂന്യ && handle->runtime_is_available != ശൂന്യ &&
                   handle->runtime_layout_info != ശൂന്യ && handle->runtime_mem_info != ശൂന്യ &&
                   handle->runtime_mem_get != ശൂന്യ && handle->runtime_mem_free != ശൂന്യ &&
                   handle->runtime_mem_try_wait != ശൂന്യ && handle->runtime_mem_release != ശൂന്യ &&
                   handle->runtime_mem_force_release != ശൂന്യ && handle->runtime_mem_query != ശൂന്യ ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    എങ്കിൽ (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == ശൂന്യ) {
        തിരികെ -1;
    }

    handle->use_global_scope = 1;
    തിരികെ spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_open_handle(സ്ഥിരം അക്ഷരം * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    സ്ഥിരം അക്ഷരം *         library  = (so_path != ശൂന്യ && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    എങ്കിൽ (resolved->module_handle != ശൂന്യ || resolved->use_global_scope) {
        തിരികെ 0;
    }

    എങ്കിൽ (spine_tcm_try_bind_global(resolved) == 0) {
        തിരികെ 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != ശൂന്യ ? 1 : 0;

    എങ്കിൽ (resolved->module_handle == ശൂന്യ) {
        spine_tcm_handle_reset(resolved);
        തിരികെ -1;
    }

    എങ്കിൽ (spine_tcm_handle_bind(resolved) != 0) {
        എങ്കിൽ (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        തിരികെ -1;
    }

    തിരികെ 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_is_available(ശൂന്യം) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ ((resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) || resolved->runtime_is_available == ശൂന്യ) {
        തിരികെ 0;
    }

    തിരികെ resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ ((resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) || resolved->runtime_layout_info == ശൂന്യ) {
        തിരികെ -1;
    }

    തിരികെ resolved->runtime_layout_info(info);
}

സ്ഥിതം വരിയിൽ സ്ഥിരം അക്ഷരം * spine_tcm_version(ശൂന്യം) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ ((resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) || resolved->runtime_version == ശൂന്യ) {
        തിരികെ "unknown";
    }

    തിരികെ resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_block_info(പൂർണ്ണസംഖ്യ id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ ((resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) || resolved->runtime_mem_info == ശൂന്യ) {
        തിരികെ -1;
    }

    തിരികെ resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
സ്ഥിതം വരിയിൽ ശൂന്യം * spine_tcm_mem_get(പൂർണ്ണസംഖ്യ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        തിരികെ ശൂന്യ;
    }

    എങ്കിൽ (resolved->runtime_mem_get == ശൂന്യ) {
        തിരികെ ശൂന്യ;
    }

    തിരികെ resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_free(പൂർണ്ണസംഖ്യ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ ((resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) || resolved->runtime_mem_free == ശൂന്യ) {
        തിരികെ -1;
    }

    തിരികെ resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
സ്ഥിതം വരിയിൽ ശൂന്യം * spine_tcm_mem_try_wait(പൂർണ്ണസംഖ്യ id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        തിരികെ ശൂന്യ;
    }

    എങ്കിൽ (resolved->runtime_mem_try_wait == ശൂന്യ) {
        തിരികെ ശൂന്യ;
    }

    തിരികെ resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_release(പൂർണ്ണസംഖ്യ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ ((resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) || resolved->runtime_mem_release == ശൂന്യ) {
        തിരികെ -1;
    }

    തിരികെ resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_force_release(പൂർണ്ണസംഖ്യ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ ((resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == ശൂന്യ) {
        തിരികെ -1;
    }

    തിരികെ resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ spine_tcm_mem_query(പൂർണ്ണസംഖ്യ id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    എങ്കിൽ (resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) {
        (ശൂന്യം) spine_tcm_open_handle(ശൂന്യ);
    }

    എങ്കിൽ ((resolved->module_handle == ശൂന്യ && !resolved->use_global_scope) || resolved->runtime_mem_query == ശൂന്യ) {
        തിരികെ -1;
    }

    തിരികെ resolved->runtime_mem_query(id);
}
#അല്ലെങ്കിൽ
സ്ഥിതം വരിയിൽ സ്ഥിരം അക്ഷരം * spine_tcm_version(ശൂന്യം) {
    തിരികെ spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
