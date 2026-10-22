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

#અન્તર્ભાવ <stddef.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <string.h>

#જો !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
બાહ્ય "C" {
#endif

#જો defined(_WIN32)
#    જો defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    નહીંતર
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#નહીંતર
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

પ્રકારવ્યાખ્યા સંરચના spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_અંક;
    પૂર્ણાંક    is_fake_tcm;
} spine_tcm_mem_info_t;

પ્રકારવ્યાખ્યા સંરચના spine_tcm_block_info {
    પૂર્ણાંક      id;
    શૂન્ય *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    પૂર્ણાંક      owner_tid;
    પૂર્ણાંક      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API અચલ અક્ષર * spine_tcm_runtime_version(શૂન્ય);
SPINE_TCM_API પૂર્ણાંક          spine_tcm_runtime_is_available(શૂન્ય);
SPINE_TCM_API પૂર્ણાંક          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API પૂર્ણાંક          spine_tcm_runtime_mem_info(પૂર્ણાંક id, spine_tcm_block_info_t * info);
SPINE_TCM_API શૂન્ય *       spine_tcm_runtime_mem_get(પૂર્ણાંક id);
SPINE_TCM_API પૂર્ણાંક          spine_tcm_runtime_mem_free(પૂર્ણાંક id);
SPINE_TCM_API શૂન્ય *       spine_tcm_runtime_mem_try_wait(પૂર્ણાંક id, size_t timeout_us);
SPINE_TCM_API પૂર્ણાંક          spine_tcm_runtime_mem_release(પૂર્ણાંક id);
SPINE_TCM_API પૂર્ણાંક          spine_tcm_runtime_mem_force_release(પૂર્ણાંક id);
SPINE_TCM_API પૂર્ણાંક          spine_tcm_runtime_mem_query(પૂર્ણાંક id);

#જો defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_open_handle(અચલ અક્ષર * so_path) {
    (શૂન્ય) so_path;
    પાછા 0;
}

સ્થિર અંતઃરેખ અચલ અક્ષર * spine_tcm_version(શૂન્ય) {
    પાછા spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_is_available(શૂન્ય) {
    પાછા spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    પાછા spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_block_info(પૂર્ણાંક id, spine_tcm_block_info_t * info) {
    પાછા spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
સ્થિર અંતઃરેખ શૂન્ય * spine_tcm_mem_get(પૂર્ણાંક id) {
    પાછા spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_free(પૂર્ણાંક id) {
    પાછા spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
સ્થિર અંતઃરેખ શૂન્ય * spine_tcm_mem_try_wait(પૂર્ણાંક id, size_t over_time) {
    પાછા spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_release(પૂર્ણાંક id) {
    પાછા spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_force_release(પૂર્ણાંક id) {
    પાછા spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_query(પૂર્ણાંક id) {
    પાછા spine_tcm_runtime_mem_query(id);
}
#અથવાજો !defined(SPINE_TCM_BUILD_SHARED)
પ્રકારવ્યાખ્યા સંરચના spine_tcm_handle {
    શૂન્ય * module_handle;
    પૂર્ણાંક    use_global_scope;
    પૂર્ણાંક    owns_module_handle;
    અચલ અક્ષર * (*runtime_version)(શૂન્ય);
    પૂર્ણાંક (*runtime_is_available)(શૂન્ય);
    પૂર્ણાંક (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    પૂર્ણાંક (*runtime_mem_info)(પૂર્ણાંક id, spine_tcm_block_info_t * info);
    શૂન્ય * (*runtime_mem_get)(પૂર્ણાંક id);
    પૂર્ણાંક (*runtime_mem_free)(પૂર્ણાંક id);
    શૂન્ય * (*runtime_mem_try_wait)(પૂર્ણાંક id, size_t over_time);
    પૂર્ણાંક (*runtime_mem_release)(પૂર્ણાંક id);
    પૂર્ણાંક (*runtime_mem_force_release)(પૂર્ણાંક id);
    પૂર્ણાંક (*runtime_mem_query)(પૂર્ણાંક id);
} spine_tcm_handle_t;

સ્થિર અંતઃરેખ spine_tcm_handle_t * spine_tcm_default_handle(શૂન્ય) {
    સ્થિર spine_tcm_handle_t handle = { 0 };
    પાછા &handle;
}

સ્થિર અંતઃરેખ શૂન્ય spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    જો (handle != ખાલી) {
        memset(handle, 0, કદ(*handle));
    }
}

સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    શૂન્ય * પ્રતીક_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (અચલ અક્ષર * (*) (શૂન્ય) ) dlsym(પ્રતીક_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (પૂર્ણાંક (*)(શૂન્ય)) dlsym(પ્રતીક_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (પૂર્ણાંક (*)(spine_tcm_mem_info_t *)) dlsym(પ્રતીક_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (પૂર્ણાંક (*)(પૂર્ણાંક, spine_tcm_block_info_t *)) dlsym(પ્રતીક_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (શૂન્ય * (*) (પૂર્ણાંક) ) dlsym(પ્રતીક_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (પૂર્ણાંક (*)(પૂર્ણાંક)) dlsym(પ્રતીક_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (શૂન્ય * (*) (પૂર્ણાંક, size_t)) dlsym(પ્રતીક_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (પૂર્ણાંક (*)(પૂર્ણાંક)) dlsym(પ્રતીક_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (પૂર્ણાંક (*)(પૂર્ણાંક)) dlsym(પ્રતીક_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (પૂર્ણાંક (*)(પૂર્ણાંક)) dlsym(પ્રતીક_scope, "spine_tcm_runtime_mem_query");

    પાછા handle->runtime_version != ખાલી && handle->runtime_is_available != ખાલી &&
                   handle->runtime_layout_info != ખાલી && handle->runtime_mem_info != ખાલી &&
                   handle->runtime_mem_get != ખાલી && handle->runtime_mem_free != ખાલી &&
                   handle->runtime_mem_try_wait != ખાલી && handle->runtime_mem_release != ખાલી &&
                   handle->runtime_mem_force_release != ખાલી && handle->runtime_mem_query != ખાલી ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    જો (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == ખાલી) {
        પાછા -1;
    }

    handle->use_global_scope = 1;
    પાછા spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_open_handle(અચલ અક્ષર * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    અચલ અક્ષર *         library  = (so_path != ખાલી && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    જો (resolved->module_handle != ખાલી || resolved->use_global_scope) {
        પાછા 0;
    }

    જો (spine_tcm_try_bind_global(resolved) == 0) {
        પાછા 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != ખાલી ? 1 : 0;

    જો (resolved->module_handle == ખાલી) {
        spine_tcm_handle_reset(resolved);
        પાછા -1;
    }

    જો (spine_tcm_handle_bind(resolved) != 0) {
        જો (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        પાછા -1;
    }

    પાછા 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_is_available(શૂન્ય) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો ((resolved->module_handle == ખાલી && !resolved->use_global_scope) || resolved->runtime_is_available == ખાલી) {
        પાછા 0;
    }

    પાછા resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો ((resolved->module_handle == ખાલી && !resolved->use_global_scope) || resolved->runtime_layout_info == ખાલી) {
        પાછા -1;
    }

    પાછા resolved->runtime_layout_info(info);
}

સ્થિર અંતઃરેખ અચલ અક્ષર * spine_tcm_version(શૂન્ય) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો ((resolved->module_handle == ખાલી && !resolved->use_global_scope) || resolved->runtime_version == ખાલી) {
        પાછા "unknown";
    }

    પાછા resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_block_info(પૂર્ણાંક id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો ((resolved->module_handle == ખાલી && !resolved->use_global_scope) || resolved->runtime_mem_info == ખાલી) {
        પાછા -1;
    }

    પાછા resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
સ્થિર અંતઃરેખ શૂન્ય * spine_tcm_mem_get(પૂર્ણાંક id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        પાછા ખાલી;
    }

    જો (resolved->runtime_mem_get == ખાલી) {
        પાછા ખાલી;
    }

    પાછા resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_free(પૂર્ણાંક id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો ((resolved->module_handle == ખાલી && !resolved->use_global_scope) || resolved->runtime_mem_free == ખાલી) {
        પાછા -1;
    }

    પાછા resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
સ્થિર અંતઃરેખ શૂન્ય * spine_tcm_mem_try_wait(પૂર્ણાંક id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        પાછા ખાલી;
    }

    જો (resolved->runtime_mem_try_wait == ખાલી) {
        પાછા ખાલી;
    }

    પાછા resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_release(પૂર્ણાંક id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો ((resolved->module_handle == ખાલી && !resolved->use_global_scope) || resolved->runtime_mem_release == ખાલી) {
        પાછા -1;
    }

    પાછા resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_force_release(પૂર્ણાંક id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો ((resolved->module_handle == ખાલી && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == ખાલી) {
        પાછા -1;
    }

    પાછા resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
સ્થિર અંતઃરેખ પૂર્ણાંક spine_tcm_mem_query(પૂર્ણાંક id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    જો (resolved->module_handle == ખાલી && !resolved->use_global_scope) {
        (શૂન્ય) spine_tcm_open_handle(ખાલી);
    }

    જો ((resolved->module_handle == ખાલી && !resolved->use_global_scope) || resolved->runtime_mem_query == ખાલી) {
        પાછા -1;
    }

    પાછા resolved->runtime_mem_query(id);
}
#નહીંતર
સ્થિર અંતઃરેખ અચલ અક્ષર * spine_tcm_version(શૂન્ય) {
    પાછા spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
