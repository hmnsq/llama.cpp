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
বাহ্যিক "C" {
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

প্রকারলেখ কাঠামো spine_tcm_mem_info {
    size_t blk_আকার;
    size_t blk_অঙ্ক;
    পূর্ণসংখ্যা    is_fake_tcm;
} spine_tcm_mem_info_t;

প্রকারলেখ কাঠামো spine_tcm_block_info {
    পূর্ণসংখ্যা      id;
    শূন্য *   va;
    size_t   আকার_2;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    পূর্ণসংখ্যা      owner_tid;
    পূর্ণসংখ্যা      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ধ্রুবক অক্ষর * spine_tcm_runtime_version(শূন্য);
SPINE_TCM_API পূর্ণসংখ্যা          spine_tcm_runtime_is_available(শূন্য);
SPINE_TCM_API পূর্ণসংখ্যা          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API পূর্ণসংখ্যা          spine_tcm_runtime_mem_info(পূর্ণসংখ্যা id, spine_tcm_block_info_t * info);
SPINE_TCM_API শূন্য *       spine_tcm_runtime_mem_get(পূর্ণসংখ্যা id);
SPINE_TCM_API পূর্ণসংখ্যা          spine_tcm_runtime_mem_free(পূর্ণসংখ্যা id);
SPINE_TCM_API শূন্য *       spine_tcm_runtime_mem_try_wait(পূর্ণসংখ্যা id, size_t timeout_us);
SPINE_TCM_API পূর্ণসংখ্যা          spine_tcm_runtime_mem_release(পূর্ণসংখ্যা id);
SPINE_TCM_API পূর্ণসংখ্যা          spine_tcm_runtime_mem_force_release(পূর্ণসংখ্যা id);
SPINE_TCM_API পূর্ণসংখ্যা          spine_tcm_runtime_mem_query(পূর্ণসংখ্যা id);

#যদি defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_open_handle(ধ্রুবক অক্ষর * so_path) {
    (শূন্য) so_path;
    ফেরত 0;
}

স্থির অন্তঃরেখ ধ্রুবক অক্ষর * spine_tcm_version(শূন্য) {
    ফেরত spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_is_available(শূন্য) {
    ফেরত spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    ফেরত spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_block_info(পূর্ণসংখ্যা id, spine_tcm_block_info_t * info) {
    ফেরত spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
স্থির অন্তঃরেখ শূন্য * spine_tcm_mem_get(পূর্ণসংখ্যা id) {
    ফেরত spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_free(পূর্ণসংখ্যা id) {
    ফেরত spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
স্থির অন্তঃরেখ শূন্য * spine_tcm_mem_try_wait(পূর্ণসংখ্যা id, size_t over_time) {
    ফেরত spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_release(পূর্ণসংখ্যা id) {
    ফেরত spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_force_release(পূর্ণসংখ্যা id) {
    ফেরত spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_query(পূর্ণসংখ্যা id) {
    ফেরত spine_tcm_runtime_mem_query(id);
}
#নতুবা_যদি !defined(SPINE_TCM_BUILD_SHARED)
প্রকারলেখ কাঠামো spine_tcm_handle {
    শূন্য * module_handle;
    পূর্ণসংখ্যা    use_global_scope;
    পূর্ণসংখ্যা    owns_module_handle;
    ধ্রুবক অক্ষর * (*runtime_version)(শূন্য);
    পূর্ণসংখ্যা (*runtime_is_available)(শূন্য);
    পূর্ণসংখ্যা (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    পূর্ণসংখ্যা (*runtime_mem_info)(পূর্ণসংখ্যা id, spine_tcm_block_info_t * info);
    শূন্য * (*runtime_mem_get)(পূর্ণসংখ্যা id);
    পূর্ণসংখ্যা (*runtime_mem_free)(পূর্ণসংখ্যা id);
    শূন্য * (*runtime_mem_try_wait)(পূর্ণসংখ্যা id, size_t over_time);
    পূর্ণসংখ্যা (*runtime_mem_release)(পূর্ণসংখ্যা id);
    পূর্ণসংখ্যা (*runtime_mem_force_release)(পূর্ণসংখ্যা id);
    পূর্ণসংখ্যা (*runtime_mem_query)(পূর্ণসংখ্যা id);
} spine_tcm_handle_t;

স্থির অন্তঃরেখ spine_tcm_handle_t * spine_tcm_default_handle(শূন্য) {
    স্থির spine_tcm_handle_t handle = { 0 };
    ফেরত &handle;
}

স্থির অন্তঃরেখ শূন্য spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    যদি (handle != ফাঁকা) {
        memset(handle, 0, আকার(*handle));
    }
}

স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    শূন্য * প্রতীক_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (ধ্রুবক অক্ষর * (*) (শূন্য) ) dlsym(প্রতীক_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (পূর্ণসংখ্যা (*)(শূন্য)) dlsym(প্রতীক_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (পূর্ণসংখ্যা (*)(spine_tcm_mem_info_t *)) dlsym(প্রতীক_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (পূর্ণসংখ্যা (*)(পূর্ণসংখ্যা, spine_tcm_block_info_t *)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (শূন্য * (*) (পূর্ণসংখ্যা) ) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (পূর্ণসংখ্যা (*)(পূর্ণসংখ্যা)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (শূন্য * (*) (পূর্ণসংখ্যা, size_t)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (পূর্ণসংখ্যা (*)(পূর্ণসংখ্যা)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (পূর্ণসংখ্যা (*)(পূর্ণসংখ্যা)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (পূর্ণসংখ্যা (*)(পূর্ণসংখ্যা)) dlsym(প্রতীক_scope, "spine_tcm_runtime_mem_query");

    ফেরত handle->runtime_version != ফাঁকা && handle->runtime_is_available != ফাঁকা &&
                   handle->runtime_layout_info != ফাঁকা && handle->runtime_mem_info != ফাঁকা &&
                   handle->runtime_mem_get != ফাঁকা && handle->runtime_mem_free != ফাঁকা &&
                   handle->runtime_mem_try_wait != ফাঁকা && handle->runtime_mem_release != ফাঁকা &&
                   handle->runtime_mem_force_release != ফাঁকা && handle->runtime_mem_query != ফাঁকা ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    যদি (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == ফাঁকা) {
        ফেরত -1;
    }

    handle->use_global_scope = 1;
    ফেরত spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_open_handle(ধ্রুবক অক্ষর * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ধ্রুবক অক্ষর *         library  = (so_path != ফাঁকা && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    যদি (resolved->module_handle != ফাঁকা || resolved->use_global_scope) {
        ফেরত 0;
    }

    যদি (spine_tcm_try_bind_global(resolved) == 0) {
        ফেরত 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != ফাঁকা ? 1 : 0;

    যদি (resolved->module_handle == ফাঁকা) {
        spine_tcm_handle_reset(resolved);
        ফেরত -1;
    }

    যদি (spine_tcm_handle_bind(resolved) != 0) {
        যদি (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        ফেরত -1;
    }

    ফেরত 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_is_available(শূন্য) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি ((resolved->module_handle == ফাঁকা && !resolved->use_global_scope) || resolved->runtime_is_available == ফাঁকা) {
        ফেরত 0;
    }

    ফেরত resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি ((resolved->module_handle == ফাঁকা && !resolved->use_global_scope) || resolved->runtime_layout_info == ফাঁকা) {
        ফেরত -1;
    }

    ফেরত resolved->runtime_layout_info(info);
}

স্থির অন্তঃরেখ ধ্রুবক অক্ষর * spine_tcm_version(শূন্য) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি ((resolved->module_handle == ফাঁকা && !resolved->use_global_scope) || resolved->runtime_version == ফাঁকা) {
        ফেরত "unknown";
    }

    ফেরত resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_block_info(পূর্ণসংখ্যা id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি ((resolved->module_handle == ফাঁকা && !resolved->use_global_scope) || resolved->runtime_mem_info == ফাঁকা) {
        ফেরত -1;
    }

    ফেরত resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
স্থির অন্তঃরেখ শূন্য * spine_tcm_mem_get(পূর্ণসংখ্যা id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        ফেরত ফাঁকা;
    }

    যদি (resolved->runtime_mem_get == ফাঁকা) {
        ফেরত ফাঁকা;
    }

    ফেরত resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_free(পূর্ণসংখ্যা id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি ((resolved->module_handle == ফাঁকা && !resolved->use_global_scope) || resolved->runtime_mem_free == ফাঁকা) {
        ফেরত -1;
    }

    ফেরত resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
স্থির অন্তঃরেখ শূন্য * spine_tcm_mem_try_wait(পূর্ণসংখ্যা id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        ফেরত ফাঁকা;
    }

    যদি (resolved->runtime_mem_try_wait == ফাঁকা) {
        ফেরত ফাঁকা;
    }

    ফেরত resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_release(পূর্ণসংখ্যা id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি ((resolved->module_handle == ফাঁকা && !resolved->use_global_scope) || resolved->runtime_mem_release == ফাঁকা) {
        ফেরত -1;
    }

    ফেরত resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_force_release(পূর্ণসংখ্যা id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি ((resolved->module_handle == ফাঁকা && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == ফাঁকা) {
        ফেরত -1;
    }

    ফেরত resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
স্থির অন্তঃরেখ পূর্ণসংখ্যা spine_tcm_mem_query(পূর্ণসংখ্যা id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    যদি (resolved->module_handle == ফাঁকা && !resolved->use_global_scope) {
        (শূন্য) spine_tcm_open_handle(ফাঁকা);
    }

    যদি ((resolved->module_handle == ফাঁকা && !resolved->use_global_scope) || resolved->runtime_mem_query == ফাঁকা) {
        ফেরত -1;
    }

    ফেরত resolved->runtime_mem_query(id);
}
#অন্যথা
স্থির অন্তঃরেখ ধ্রুবক অক্ষর * spine_tcm_version(শূন্য) {
    ফেরত spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
