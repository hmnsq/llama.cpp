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

#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <string.h>

#যদি !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
বাহ্য "C" {
#endif

#যদি defined(_WIN32)
#    যদি defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    অন্যথা
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#অন্যথা
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

প্রকারনির্দেশ সংরচনা spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_অঙ্ক;
    পূর্ণাঙ্ক    is_fake_tcm;
} spine_tcm_mem_info_t;

প্রকারনির্দেশ সংরচনা spine_tcm_block_info {
    পূর্ণাঙ্ক      id;
    শূন্য *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    পূর্ণাঙ্ক      owner_tid;
    পূর্ণাঙ্ক      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ধ্রুব বর্ণ * spine_tcm_runtime_version(শূন্য);
SPINE_TCM_API পূর্ণাঙ্ক          spine_tcm_runtime_is_available(শূন্য);
SPINE_TCM_API পূর্ণাঙ্ক          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API পূর্ণাঙ্ক          spine_tcm_runtime_mem_info(পূর্ণাঙ্ক id, spine_tcm_block_info_t * info);
SPINE_TCM_API শূন্য *       spine_tcm_runtime_mem_get(পূর্ণাঙ্ক id);
SPINE_TCM_API পূর্ণাঙ্ক          spine_tcm_runtime_mem_free(পূর্ণাঙ্ক id);
SPINE_TCM_API শূন্য *       spine_tcm_runtime_mem_try_wait(পূর্ণাঙ্ক id, size_t timeout_us);
SPINE_TCM_API পূর্ণাঙ্ক          spine_tcm_runtime_mem_release(পূর্ণাঙ্ক id);
SPINE_TCM_API পূর্ণাঙ্ক          spine_tcm_runtime_mem_force_release(পূর্ণাঙ্ক id);
SPINE_TCM_API পূর্ণাঙ্ক          spine_tcm_runtime_mem_query(পূর্ণাঙ্ক id);

#যদি defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_open_handle(ধ্রুব বর্ণ * so_path) {
    (শূন্য) so_path;
    প্রত্যাগম 0;
}

স্থির অন্তঃস্থ ধ্রুব বর্ণ * spine_tcm_version(শূন্য) {
    প্রত্যাগম spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_is_available(শূন্য) {
    প্রত্যাগম spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    প্রত্যাগম spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_block_info(পূর্ণাঙ্ক id, spine_tcm_block_info_t * info) {
    প্রত্যাগম spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
স্থির অন্তঃস্থ শূন্য * spine_tcm_mem_get(পূর্ণাঙ্ক id) {
    প্রত্যাগম spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_free(পূর্ণাঙ্ক id) {
    প্রত্যাগম spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
স্থির অন্তঃস্থ শূন্য * spine_tcm_mem_try_wait(পূর্ণাঙ্ক id, size_t over_time) {
    প্রত্যাগম spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_release(পূর্ণাঙ্ক id) {
    প্রত্যাগম spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_force_release(পূর্ণাঙ্ক id) {
    প্রত্যাগম spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_query(পূর্ণাঙ্ক id) {
    প্রত্যাগম spine_tcm_runtime_mem_query(id);
}
#নোচেদ্যদি !defined(SPINE_TCM_BUILD_SHARED)
প্রকারনির্দেশ সংরচনা spine_tcm_handle {
    শূন্য * module_handle;
    পূর্ণাঙ্ক    use_global_scope;
    পূর্ণাঙ্ক    owns_module_handle;
    ধ্রুব বর্ণ * (*runtime_version)(শূন্য);
    পূর্ণাঙ্ক (*runtime_is_available)(শূন্য);
    পূর্ণাঙ্ক (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    পূর্ণাঙ্ক (*runtime_mem_info)(পূর্ণাঙ্ক id, spine_tcm_block_info_t * info);
    শূন্য * (*runtime_mem_get)(পূর্ণাঙ্ক id);
    পূর্ণাঙ্ক (*runtime_mem_free)(পূর্ণাঙ্ক id);
    শূন্য * (*runtime_mem_try_wait)(পূর্ণাঙ্ক id, size_t over_time);
    পূর্ণাঙ্ক (*runtime_mem_release)(পূর্ণাঙ্ক id);
    পূর্ণাঙ্ক (*runtime_mem_force_release)(পূর্ণাঙ্ক id);
    পূর্ণাঙ্ক (*runtime_mem_query)(পূর্ণাঙ্ক id);
} spine_tcm_handle_t;

স্থির অন্তঃস্থ spine_tcm_handle_t * spine_tcm_default_handle(শূন্য) {
    স্থির spine_tcm_handle_t handle = { 0 };
    প্রত্যাগম &handle;
}

স্থির অন্তঃস্থ শূন্য spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    যদি (handle != রিক্ত) {
        memset(handle, 0, পরিমাণ(*handle));
    }
}

স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    শূন্য * প্রতীক_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (ধ্রুব বর্ণ * (*) (শূন্য) ) dlsym(প্রতীক_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (পূর্ণাঙ্ক (*)(শূন্য)) dlsym(প্রতীক_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (পূর্ণাঙ্ক (*)(spine_tcm_mem_info_t *)) dlsym(প্রতীক_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (পূর্ণাঙ্ক (*)(পূর্ণাঙ্ক, spine_tcm_block_info_t *)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (শূন্য * (*) (পূর্ণাঙ্ক) ) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (পূর্ণাঙ্ক (*)(পূর্ণাঙ্ক)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (শূন্য * (*) (পূর্ণাঙ্ক, size_t)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (পূর্ণাঙ্ক (*)(পূর্ণাঙ্ক)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (পূর্ণাঙ্ক (*)(পূর্ণাঙ্ক)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (পূর্ণাঙ্ক (*)(পূর্ণাঙ্ক)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_query");

    প্রত্যাগম handle->runtime_version != রিক্ত && handle->runtime_is_available != রিক্ত &&
                   handle->runtime_layout_info != রিক্ত && handle->runtime_mem_info != রিক্ত &&
                   handle->runtime_mem_get != রিক্ত && handle->runtime_mem_free != রিক্ত &&
                   handle->runtime_mem_try_wait != রিক্ত && handle->runtime_mem_release != রিক্ত &&
                   handle->runtime_mem_force_release != রিক্ত && handle->runtime_mem_query != রিক্ত ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    যদি (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == রিক্ত) {
        প্রত্যাগম -1;
    }

    handle->use_global_scope = 1;
    প্রত্যাগম spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_open_handle(ধ্রুব বর্ণ * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ধ্রুব বর্ণ *         library  = (so_path != রিক্ত && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    যদি (resolved->module_handle != রিক্ত || resolved->use_global_scope) {
        প্রত্যাগম 0;
    }

    যদি (spine_tcm_try_bind_global(resolved) == 0) {
        প্রত্যাগম 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != রিক্ত ? 1 : 0;

    যদি (resolved->module_handle == রিক্ত) {
        spine_tcm_handle_reset(resolved);
        প্রত্যাগম -1;
    }

    যদি (spine_tcm_handle_bind(resolved) != 0) {
        যদি (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        প্রত্যাগম -1;
    }

    প্রত্যাগম 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_is_available(শূন্য) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি ((resolved->module_handle == রিক্ত && !resolved->use_global_scope) || resolved->runtime_is_available == রিক্ত) {
        প্রত্যাগম 0;
    }

    প্রত্যাগম resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি ((resolved->module_handle == রিক্ত && !resolved->use_global_scope) || resolved->runtime_layout_info == রিক্ত) {
        প্রত্যাগম -1;
    }

    প্রত্যাগম resolved->runtime_layout_info(info);
}

স্থির অন্তঃস্থ ধ্রুব বর্ণ * spine_tcm_version(শূন্য) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি ((resolved->module_handle == রিক্ত && !resolved->use_global_scope) || resolved->runtime_version == রিক্ত) {
        প্রত্যাগম "unknown";
    }

    প্রত্যাগম resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_block_info(পূর্ণাঙ্ক id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি ((resolved->module_handle == রিক্ত && !resolved->use_global_scope) || resolved->runtime_mem_info == রিক্ত) {
        প্রত্যাগম -1;
    }

    প্রত্যাগম resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
স্থির অন্তঃস্থ শূন্য * spine_tcm_mem_get(পূর্ণাঙ্ক id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        প্রত্যাগম রিক্ত;
    }

    যদি (resolved->runtime_mem_get == রিক্ত) {
        প্রত্যাগম রিক্ত;
    }

    প্রত্যাগম resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_free(পূর্ণাঙ্ক id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি ((resolved->module_handle == রিক্ত && !resolved->use_global_scope) || resolved->runtime_mem_free == রিক্ত) {
        প্রত্যাগম -1;
    }

    প্রত্যাগম resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
স্থির অন্তঃস্থ শূন্য * spine_tcm_mem_try_wait(পূর্ণাঙ্ক id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        প্রত্যাগম রিক্ত;
    }

    যদি (resolved->runtime_mem_try_wait == রিক্ত) {
        প্রত্যাগম রিক্ত;
    }

    প্রত্যাগম resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_release(পূর্ণাঙ্ক id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি ((resolved->module_handle == রিক্ত && !resolved->use_global_scope) || resolved->runtime_mem_release == রিক্ত) {
        প্রত্যাগম -1;
    }

    প্রত্যাগম resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_force_release(পূর্ণাঙ্ক id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি ((resolved->module_handle == রিক্ত && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == রিক্ত) {
        প্রত্যাগম -1;
    }

    প্রত্যাগম resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
স্থির অন্তঃস্থ পূর্ণাঙ্ক spine_tcm_mem_query(পূর্ণাঙ্ক id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == রিক্ত && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(রিক্ত);
    }

    যদি ((resolved->module_handle == রিক্ত && !resolved->use_global_scope) || resolved->runtime_mem_query == রিক্ত) {
        প্রত্যাগম -1;
    }

    প্রত্যাগম resolved->runtime_mem_query(id);
}
#অন্যথা
স্থির অন্তঃস্থ ধ্রুব বর্ণ * spine_tcm_version(শূন্য) {
    প্রত্যাগম spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
