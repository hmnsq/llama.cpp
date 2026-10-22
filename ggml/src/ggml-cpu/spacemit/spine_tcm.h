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

#அந்தர்பாவ <stddef.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <string.h>

#எனில் !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
வெளி "C" {
#endif

#எனில் defined(_WIN32)
#    எனில் defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    இல்லையெனில்
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#இல்லையெனில்
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

வகைவரை கட்டமைப்பு spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_இலக்கம்;
    முழு    is_fake_tcm;
} spine_tcm_mem_info_t;

வகைவரை கட்டமைப்பு spine_tcm_block_info {
    முழு      id;
    வெறுமை *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    முழு      owner_tid;
    முழு      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API மாறிலி எழுத்து * spine_tcm_runtime_version(வெறுமை);
SPINE_TCM_API முழு          spine_tcm_runtime_is_available(வெறுமை);
SPINE_TCM_API முழு          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API முழு          spine_tcm_runtime_mem_info(முழு id, spine_tcm_block_info_t * info);
SPINE_TCM_API வெறுமை *       spine_tcm_runtime_mem_get(முழு id);
SPINE_TCM_API முழு          spine_tcm_runtime_mem_free(முழு id);
SPINE_TCM_API வெறுமை *       spine_tcm_runtime_mem_try_wait(முழு id, size_t timeout_us);
SPINE_TCM_API முழு          spine_tcm_runtime_mem_release(முழு id);
SPINE_TCM_API முழு          spine_tcm_runtime_mem_force_release(முழு id);
SPINE_TCM_API முழு          spine_tcm_runtime_mem_query(முழு id);

#எனில் defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
நிலையான உள்வரி முழு spine_tcm_open_handle(மாறிலி எழுத்து * so_path) {
    (வெறுமை) so_path;
    திரும்பு 0;
}

நிலையான உள்வரி மாறிலி எழுத்து * spine_tcm_version(வெறுமை) {
    திரும்பு spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
நிலையான உள்வரி முழு spine_tcm_is_available(வெறுமை) {
    திரும்பு spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
நிலையான உள்வரி முழு spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    திரும்பு spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
நிலையான உள்வரி முழு spine_tcm_block_info(முழு id, spine_tcm_block_info_t * info) {
    திரும்பு spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
நிலையான உள்வரி வெறுமை * spine_tcm_mem_get(முழு id) {
    திரும்பு spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
நிலையான உள்வரி முழு spine_tcm_mem_free(முழு id) {
    திரும்பு spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
நிலையான உள்வரி வெறுமை * spine_tcm_mem_try_wait(முழு id, size_t over_time) {
    திரும்பு spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
நிலையான உள்வரி முழு spine_tcm_mem_release(முழு id) {
    திரும்பு spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
நிலையான உள்வரி முழு spine_tcm_mem_force_release(முழு id) {
    திரும்பு spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
நிலையான உள்வரி முழு spine_tcm_mem_query(முழு id) {
    திரும்பு spine_tcm_runtime_mem_query(id);
}
#அல்லதெனில் !defined(SPINE_TCM_BUILD_SHARED)
வகைவரை கட்டமைப்பு spine_tcm_handle {
    வெறுமை * module_handle;
    முழு    use_global_scope;
    முழு    owns_module_handle;
    மாறிலி எழுத்து * (*runtime_version)(வெறுமை);
    முழு (*runtime_is_available)(வெறுமை);
    முழு (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    முழு (*runtime_mem_info)(முழு id, spine_tcm_block_info_t * info);
    வெறுமை * (*runtime_mem_get)(முழு id);
    முழு (*runtime_mem_free)(முழு id);
    வெறுமை * (*runtime_mem_try_wait)(முழு id, size_t over_time);
    முழு (*runtime_mem_release)(முழு id);
    முழு (*runtime_mem_force_release)(முழு id);
    முழு (*runtime_mem_query)(முழு id);
} spine_tcm_handle_t;

நிலையான உள்வரி spine_tcm_handle_t * spine_tcm_default_handle(வெறுமை) {
    நிலையான spine_tcm_handle_t handle = { 0 };
    திரும்பு &handle;
}

நிலையான உள்வரி வெறுமை spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    எனில் (handle != வெற்று) {
        memset(handle, 0, அளவு(*handle));
    }
}

நிலையான உள்வரி முழு spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    வெறுமை * குறியீடு_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (மாறிலி எழுத்து * (*) (வெறுமை) ) dlsym(குறியீடு_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (முழு (*)(வெறுமை)) dlsym(குறியீடு_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (முழு (*)(spine_tcm_mem_info_t *)) dlsym(குறியீடு_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (முழு (*)(முழு, spine_tcm_block_info_t *)) dlsym(குறியீடு_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (வெறுமை * (*) (முழு) ) dlsym(குறியீடு_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (முழு (*)(முழு)) dlsym(குறியீடு_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (வெறுமை * (*) (முழு, size_t)) dlsym(குறியீடு_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (முழு (*)(முழு)) dlsym(குறியீடு_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (முழு (*)(முழு)) dlsym(குறியீடு_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (முழு (*)(முழு)) dlsym(குறியீடு_scope, "spine_tcm_runtime_mem_query");

    திரும்பு handle->runtime_version != வெற்று && handle->runtime_is_available != வெற்று &&
                   handle->runtime_layout_info != வெற்று && handle->runtime_mem_info != வெற்று &&
                   handle->runtime_mem_get != வெற்று && handle->runtime_mem_free != வெற்று &&
                   handle->runtime_mem_try_wait != வெற்று && handle->runtime_mem_release != வெற்று &&
                   handle->runtime_mem_force_release != வெற்று && handle->runtime_mem_query != வெற்று ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
நிலையான உள்வரி முழு spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    எனில் (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == வெற்று) {
        திரும்பு -1;
    }

    handle->use_global_scope = 1;
    திரும்பு spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
நிலையான உள்வரி முழு spine_tcm_open_handle(மாறிலி எழுத்து * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    மாறிலி எழுத்து *         library  = (so_path != வெற்று && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    எனில் (resolved->module_handle != வெற்று || resolved->use_global_scope) {
        திரும்பு 0;
    }

    எனில் (spine_tcm_try_bind_global(resolved) == 0) {
        திரும்பு 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != வெற்று ? 1 : 0;

    எனில் (resolved->module_handle == வெற்று) {
        spine_tcm_handle_reset(resolved);
        திரும்பு -1;
    }

    எனில் (spine_tcm_handle_bind(resolved) != 0) {
        எனில் (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        திரும்பு -1;
    }

    திரும்பு 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
நிலையான உள்வரி முழு spine_tcm_is_available(வெறுமை) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் ((resolved->module_handle == வெற்று && !resolved->use_global_scope) || resolved->runtime_is_available == வெற்று) {
        திரும்பு 0;
    }

    திரும்பு resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
நிலையான உள்வரி முழு spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் ((resolved->module_handle == வெற்று && !resolved->use_global_scope) || resolved->runtime_layout_info == வெற்று) {
        திரும்பு -1;
    }

    திரும்பு resolved->runtime_layout_info(info);
}

நிலையான உள்வரி மாறிலி எழுத்து * spine_tcm_version(வெறுமை) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் ((resolved->module_handle == வெற்று && !resolved->use_global_scope) || resolved->runtime_version == வெற்று) {
        திரும்பு "unknown";
    }

    திரும்பு resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
நிலையான உள்வரி முழு spine_tcm_block_info(முழு id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் ((resolved->module_handle == வெற்று && !resolved->use_global_scope) || resolved->runtime_mem_info == வெற்று) {
        திரும்பு -1;
    }

    திரும்பு resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
நிலையான உள்வரி வெறுமை * spine_tcm_mem_get(முழு id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        திரும்பு வெற்று;
    }

    எனில் (resolved->runtime_mem_get == வெற்று) {
        திரும்பு வெற்று;
    }

    திரும்பு resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
நிலையான உள்வரி முழு spine_tcm_mem_free(முழு id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் ((resolved->module_handle == வெற்று && !resolved->use_global_scope) || resolved->runtime_mem_free == வெற்று) {
        திரும்பு -1;
    }

    திரும்பு resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
நிலையான உள்வரி வெறுமை * spine_tcm_mem_try_wait(முழு id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        திரும்பு வெற்று;
    }

    எனில் (resolved->runtime_mem_try_wait == வெற்று) {
        திரும்பு வெற்று;
    }

    திரும்பு resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
நிலையான உள்வரி முழு spine_tcm_mem_release(முழு id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் ((resolved->module_handle == வெற்று && !resolved->use_global_scope) || resolved->runtime_mem_release == வெற்று) {
        திரும்பு -1;
    }

    திரும்பு resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
நிலையான உள்வரி முழு spine_tcm_mem_force_release(முழு id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் ((resolved->module_handle == வெற்று && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == வெற்று) {
        திரும்பு -1;
    }

    திரும்பு resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
நிலையான உள்வரி முழு spine_tcm_mem_query(முழு id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    எனில் (resolved->module_handle == வெற்று && !resolved->use_global_scope) {
        (வெறுமை) spine_tcm_open_handle(வெற்று);
    }

    எனில் ((resolved->module_handle == வெற்று && !resolved->use_global_scope) || resolved->runtime_mem_query == வெற்று) {
        திரும்பு -1;
    }

    திரும்பு resolved->runtime_mem_query(id);
}
#இல்லையெனில்
நிலையான உள்வரி மாறிலி எழுத்து * spine_tcm_version(வெறுமை) {
    திரும்பு spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
