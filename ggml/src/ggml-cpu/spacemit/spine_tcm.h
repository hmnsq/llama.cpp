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

#شامل <stddef.h>
#شامل <stdint.h>
#شامل <string.h>

#اگر !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
بیرونی "C" {
#endif

#اگر defined(_WIN32)
#    اگر defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    وگرنه
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#وگرنه
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

تعریفنوع ساختار spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_رقم;
    صحیح    is_fake_tcm;
} spine_tcm_mem_info_t;

تعریفنوع ساختار spine_tcm_block_info {
    صحیح      id;
    تهی *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    صحیح      owner_tid;
    صحیح      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ثابت نویسه * spine_tcm_runtime_version(تهی);
SPINE_TCM_API صحیح          spine_tcm_runtime_is_available(تهی);
SPINE_TCM_API صحیح          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_info(صحیح id, spine_tcm_block_info_t * info);
SPINE_TCM_API تهی *       spine_tcm_runtime_mem_get(صحیح id);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_free(صحیح id);
SPINE_TCM_API تهی *       spine_tcm_runtime_mem_try_wait(صحیح id, size_t timeout_us);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_release(صحیح id);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_force_release(صحیح id);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_query(صحیح id);

#اگر defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
ایستا درونخطی صحیح spine_tcm_open_handle(ثابت نویسه * so_path) {
    (تهی) so_path;
    بازگشت 0;
}

ایستا درونخطی ثابت نویسه * spine_tcm_version(تهی) {
    بازگشت spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ایستا درونخطی صحیح spine_tcm_is_available(تهی) {
    بازگشت spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ایستا درونخطی صحیح spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    بازگشت spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
ایستا درونخطی صحیح spine_tcm_block_info(صحیح id, spine_tcm_block_info_t * info) {
    بازگشت spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ایستا درونخطی تهی * spine_tcm_mem_get(صحیح id) {
    بازگشت spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ایستا درونخطی صحیح spine_tcm_mem_free(صحیح id) {
    بازگشت spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ایستا درونخطی تهی * spine_tcm_mem_try_wait(صحیح id, size_t over_time) {
    بازگشت spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ایستا درونخطی صحیح spine_tcm_mem_release(صحیح id) {
    بازگشت spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ایستا درونخطی صحیح spine_tcm_mem_force_release(صحیح id) {
    بازگشت spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ایستا درونخطی صحیح spine_tcm_mem_query(صحیح id) {
    بازگشت spine_tcm_runtime_mem_query(id);
}
#وگرنهاگر !defined(SPINE_TCM_BUILD_SHARED)
تعریفنوع ساختار spine_tcm_handle {
    تهی * module_handle;
    صحیح    use_global_scope;
    صحیح    owns_module_handle;
    ثابت نویسه * (*runtime_version)(تهی);
    صحیح (*runtime_is_available)(تهی);
    صحیح (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    صحیح (*runtime_mem_info)(صحیح id, spine_tcm_block_info_t * info);
    تهی * (*runtime_mem_get)(صحیح id);
    صحیح (*runtime_mem_free)(صحیح id);
    تهی * (*runtime_mem_try_wait)(صحیح id, size_t over_time);
    صحیح (*runtime_mem_release)(صحیح id);
    صحیح (*runtime_mem_force_release)(صحیح id);
    صحیح (*runtime_mem_query)(صحیح id);
} spine_tcm_handle_t;

ایستا درونخطی spine_tcm_handle_t * spine_tcm_default_handle(تهی) {
    ایستا spine_tcm_handle_t handle = { 0 };
    بازگشت &handle;
}

ایستا درونخطی تهی spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    اگر (handle != پوچ) {
        memset(handle, 0, اندازه(*handle));
    }
}

ایستا درونخطی صحیح spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    تهی * نماد_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (ثابت نویسه * (*) (تهی) ) dlsym(نماد_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (صحیح (*)(تهی)) dlsym(نماد_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (صحیح (*)(spine_tcm_mem_info_t *)) dlsym(نماد_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (صحیح (*)(صحیح, spine_tcm_block_info_t *)) dlsym(نماد_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (تهی * (*) (صحیح) ) dlsym(نماد_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (صحیح (*)(صحیح)) dlsym(نماد_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (تهی * (*) (صحیح, size_t)) dlsym(نماد_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (صحیح (*)(صحیح)) dlsym(نماد_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (صحیح (*)(صحیح)) dlsym(نماد_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (صحیح (*)(صحیح)) dlsym(نماد_scope, "spine_tcm_runtime_mem_query");

    بازگشت handle->runtime_version != پوچ && handle->runtime_is_available != پوچ &&
                   handle->runtime_layout_info != پوچ && handle->runtime_mem_info != پوچ &&
                   handle->runtime_mem_get != پوچ && handle->runtime_mem_free != پوچ &&
                   handle->runtime_mem_try_wait != پوچ && handle->runtime_mem_release != پوچ &&
                   handle->runtime_mem_force_release != پوچ && handle->runtime_mem_query != پوچ ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
ایستا درونخطی صحیح spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    اگر (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == پوچ) {
        بازگشت -1;
    }

    handle->use_global_scope = 1;
    بازگشت spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
ایستا درونخطی صحیح spine_tcm_open_handle(ثابت نویسه * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ثابت نویسه *         library  = (so_path != پوچ && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    اگر (resolved->module_handle != پوچ || resolved->use_global_scope) {
        بازگشت 0;
    }

    اگر (spine_tcm_try_bind_global(resolved) == 0) {
        بازگشت 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != پوچ ? 1 : 0;

    اگر (resolved->module_handle == پوچ) {
        spine_tcm_handle_reset(resolved);
        بازگشت -1;
    }

    اگر (spine_tcm_handle_bind(resolved) != 0) {
        اگر (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        بازگشت -1;
    }

    بازگشت 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ایستا درونخطی صحیح spine_tcm_is_available(تهی) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر ((resolved->module_handle == پوچ && !resolved->use_global_scope) || resolved->runtime_is_available == پوچ) {
        بازگشت 0;
    }

    بازگشت resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ایستا درونخطی صحیح spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر ((resolved->module_handle == پوچ && !resolved->use_global_scope) || resolved->runtime_layout_info == پوچ) {
        بازگشت -1;
    }

    بازگشت resolved->runtime_layout_info(info);
}

ایستا درونخطی ثابت نویسه * spine_tcm_version(تهی) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر ((resolved->module_handle == پوچ && !resolved->use_global_scope) || resolved->runtime_version == پوچ) {
        بازگشت "unknown";
    }

    بازگشت resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
ایستا درونخطی صحیح spine_tcm_block_info(صحیح id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر ((resolved->module_handle == پوچ && !resolved->use_global_scope) || resolved->runtime_mem_info == پوچ) {
        بازگشت -1;
    }

    بازگشت resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ایستا درونخطی تهی * spine_tcm_mem_get(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        بازگشت پوچ;
    }

    اگر (resolved->runtime_mem_get == پوچ) {
        بازگشت پوچ;
    }

    بازگشت resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ایستا درونخطی صحیح spine_tcm_mem_free(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر ((resolved->module_handle == پوچ && !resolved->use_global_scope) || resolved->runtime_mem_free == پوچ) {
        بازگشت -1;
    }

    بازگشت resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ایستا درونخطی تهی * spine_tcm_mem_try_wait(صحیح id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        بازگشت پوچ;
    }

    اگر (resolved->runtime_mem_try_wait == پوچ) {
        بازگشت پوچ;
    }

    بازگشت resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ایستا درونخطی صحیح spine_tcm_mem_release(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر ((resolved->module_handle == پوچ && !resolved->use_global_scope) || resolved->runtime_mem_release == پوچ) {
        بازگشت -1;
    }

    بازگشت resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ایستا درونخطی صحیح spine_tcm_mem_force_release(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر ((resolved->module_handle == پوچ && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == پوچ) {
        بازگشت -1;
    }

    بازگشت resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ایستا درونخطی صحیح spine_tcm_mem_query(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == پوچ && !resolved->use_global_scope) {
        (تهی) spine_tcm_open_handle(پوچ);
    }

    اگر ((resolved->module_handle == پوچ && !resolved->use_global_scope) || resolved->runtime_mem_query == پوچ) {
        بازگشت -1;
    }

    بازگشت resolved->runtime_mem_query(id);
}
#وگرنه
ایستا درونخطی ثابت نویسه * spine_tcm_version(تهی) {
    بازگشت spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
