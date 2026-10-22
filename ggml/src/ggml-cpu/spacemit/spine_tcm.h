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

#ضمن <stddef.h>
#ضمن <stdint.h>
#ضمن <string.h>

#إذا !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
خارجي "C" {
#endif

#إذا defined(_WIN32)
#    إذا defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    وإلا
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#وإلا
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

تعريفنوع بنية spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_اڠک;
    صحيح    is_fake_tcm;
} spine_tcm_mem_info_t;

تعريفنوع بنية spine_tcm_block_info {
    صحيح      id;
    فراغ *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    صحيح      owner_tid;
    صحيح      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ثابت حرف * spine_tcm_runtime_version(فراغ);
SPINE_TCM_API صحيح          spine_tcm_runtime_is_available(فراغ);
SPINE_TCM_API صحيح          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API صحيح          spine_tcm_runtime_mem_info(صحيح id, spine_tcm_block_info_t * info);
SPINE_TCM_API فراغ *       spine_tcm_runtime_mem_get(صحيح id);
SPINE_TCM_API صحيح          spine_tcm_runtime_mem_free(صحيح id);
SPINE_TCM_API فراغ *       spine_tcm_runtime_mem_try_wait(صحيح id, size_t timeout_us);
SPINE_TCM_API صحيح          spine_tcm_runtime_mem_release(صحيح id);
SPINE_TCM_API صحيح          spine_tcm_runtime_mem_force_release(صحيح id);
SPINE_TCM_API صحيح          spine_tcm_runtime_mem_query(صحيح id);

#إذا defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
ساكن مضمن صحيح spine_tcm_open_handle(ثابت حرف * so_path) {
    (فراغ) so_path;
    أرجع 0;
}

ساكن مضمن ثابت حرف * spine_tcm_version(فراغ) {
    أرجع spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ساكن مضمن صحيح spine_tcm_is_available(فراغ) {
    أرجع spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ساكن مضمن صحيح spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    أرجع spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
ساكن مضمن صحيح spine_tcm_block_info(صحيح id, spine_tcm_block_info_t * info) {
    أرجع spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ساكن مضمن فراغ * spine_tcm_mem_get(صحيح id) {
    أرجع spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ساكن مضمن صحيح spine_tcm_mem_free(صحيح id) {
    أرجع spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ساكن مضمن فراغ * spine_tcm_mem_try_wait(صحيح id, size_t over_time) {
    أرجع spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ساكن مضمن صحيح spine_tcm_mem_release(صحيح id) {
    أرجع spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ساكن مضمن صحيح spine_tcm_mem_force_release(صحيح id) {
    أرجع spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ساكن مضمن صحيح spine_tcm_mem_query(صحيح id) {
    أرجع spine_tcm_runtime_mem_query(id);
}
#وإلاإذا !defined(SPINE_TCM_BUILD_SHARED)
تعريفنوع بنية spine_tcm_handle {
    فراغ * module_handle;
    صحيح    use_global_scope;
    صحيح    owns_module_handle;
    ثابت حرف * (*runtime_version)(فراغ);
    صحيح (*runtime_is_available)(فراغ);
    صحيح (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    صحيح (*runtime_mem_info)(صحيح id, spine_tcm_block_info_t * info);
    فراغ * (*runtime_mem_get)(صحيح id);
    صحيح (*runtime_mem_free)(صحيح id);
    فراغ * (*runtime_mem_try_wait)(صحيح id, size_t over_time);
    صحيح (*runtime_mem_release)(صحيح id);
    صحيح (*runtime_mem_force_release)(صحيح id);
    صحيح (*runtime_mem_query)(صحيح id);
} spine_tcm_handle_t;

ساكن مضمن spine_tcm_handle_t * spine_tcm_default_handle(فراغ) {
    ساكن spine_tcm_handle_t handle = { 0 };
    أرجع &handle;
}

ساكن مضمن فراغ spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    إذا (handle != عدم) {
        memset(handle, 0, حجم(*handle));
    }
}

ساكن مضمن صحيح spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    فراغ * سيمبول_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (ثابت حرف * (*) (فراغ) ) dlsym(سيمبول_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (صحيح (*)(فراغ)) dlsym(سيمبول_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (صحيح (*)(spine_tcm_mem_info_t *)) dlsym(سيمبول_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (صحيح (*)(صحيح, spine_tcm_block_info_t *)) dlsym(سيمبول_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (فراغ * (*) (صحيح) ) dlsym(سيمبول_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (صحيح (*)(صحيح)) dlsym(سيمبول_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (فراغ * (*) (صحيح, size_t)) dlsym(سيمبول_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (صحيح (*)(صحيح)) dlsym(سيمبول_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (صحيح (*)(صحيح)) dlsym(سيمبول_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (صحيح (*)(صحيح)) dlsym(سيمبول_scope, "spine_tcm_runtime_mem_query");

    أرجع handle->runtime_version != عدم && handle->runtime_is_available != عدم &&
                   handle->runtime_layout_info != عدم && handle->runtime_mem_info != عدم &&
                   handle->runtime_mem_get != عدم && handle->runtime_mem_free != عدم &&
                   handle->runtime_mem_try_wait != عدم && handle->runtime_mem_release != عدم &&
                   handle->runtime_mem_force_release != عدم && handle->runtime_mem_query != عدم ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
ساكن مضمن صحيح spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    إذا (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == عدم) {
        أرجع -1;
    }

    handle->use_global_scope = 1;
    أرجع spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
ساكن مضمن صحيح spine_tcm_open_handle(ثابت حرف * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ثابت حرف *         library  = (so_path != عدم && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    إذا (resolved->module_handle != عدم || resolved->use_global_scope) {
        أرجع 0;
    }

    إذا (spine_tcm_try_bind_global(resolved) == 0) {
        أرجع 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != عدم ? 1 : 0;

    إذا (resolved->module_handle == عدم) {
        spine_tcm_handle_reset(resolved);
        أرجع -1;
    }

    إذا (spine_tcm_handle_bind(resolved) != 0) {
        إذا (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        أرجع -1;
    }

    أرجع 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ساكن مضمن صحيح spine_tcm_is_available(فراغ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا ((resolved->module_handle == عدم && !resolved->use_global_scope) || resolved->runtime_is_available == عدم) {
        أرجع 0;
    }

    أرجع resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ساكن مضمن صحيح spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا ((resolved->module_handle == عدم && !resolved->use_global_scope) || resolved->runtime_layout_info == عدم) {
        أرجع -1;
    }

    أرجع resolved->runtime_layout_info(info);
}

ساكن مضمن ثابت حرف * spine_tcm_version(فراغ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا ((resolved->module_handle == عدم && !resolved->use_global_scope) || resolved->runtime_version == عدم) {
        أرجع "unknown";
    }

    أرجع resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
ساكن مضمن صحيح spine_tcm_block_info(صحيح id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا ((resolved->module_handle == عدم && !resolved->use_global_scope) || resolved->runtime_mem_info == عدم) {
        أرجع -1;
    }

    أرجع resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ساكن مضمن فراغ * spine_tcm_mem_get(صحيح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        أرجع عدم;
    }

    إذا (resolved->runtime_mem_get == عدم) {
        أرجع عدم;
    }

    أرجع resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ساكن مضمن صحيح spine_tcm_mem_free(صحيح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا ((resolved->module_handle == عدم && !resolved->use_global_scope) || resolved->runtime_mem_free == عدم) {
        أرجع -1;
    }

    أرجع resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ساكن مضمن فراغ * spine_tcm_mem_try_wait(صحيح id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        أرجع عدم;
    }

    إذا (resolved->runtime_mem_try_wait == عدم) {
        أرجع عدم;
    }

    أرجع resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ساكن مضمن صحيح spine_tcm_mem_release(صحيح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا ((resolved->module_handle == عدم && !resolved->use_global_scope) || resolved->runtime_mem_release == عدم) {
        أرجع -1;
    }

    أرجع resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ساكن مضمن صحيح spine_tcm_mem_force_release(صحيح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا ((resolved->module_handle == عدم && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == عدم) {
        أرجع -1;
    }

    أرجع resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ساكن مضمن صحيح spine_tcm_mem_query(صحيح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    إذا (resolved->module_handle == عدم && !resolved->use_global_scope) {
        (فراغ) spine_tcm_open_handle(عدم);
    }

    إذا ((resolved->module_handle == عدم && !resolved->use_global_scope) || resolved->runtime_mem_query == عدم) {
        أرجع -1;
    }

    أرجع resolved->runtime_mem_query(id);
}
#وإلا
ساكن مضمن ثابت حرف * spine_tcm_version(فراغ) {
    أرجع spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
