#నిర్వచితం_కాకపోతే SPINE_TCM_ప్రజా_H
#నిర్వచించు SPINE_TCM_ప్రజా_H

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

#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>

#ఒకవేళ !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_ప్రత్యక్ష_LINK)
#    include <dlfcn.h>
#అయితే_ముగింపు

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

#ఒకవేళ defined(_WIN32)
#    ఒకవేళ defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    లేకపోతే
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#లేకపోతే
#    define SPINE_TCM_API __attribute__((visibility("default")))
#అయితే_ముగింపు

రకనిర్వచనం నిర్మాణం spine_tcm_mem_info {
    size_t blk_కొలత;
    size_t blk_అంకె;
    పూర్ణాంకం    is_fake_tcm;
} spine_tcm_mem_info_t;

రకనిర్వచనం నిర్మాణం spine_tcm_ఖండం_info {
    పూర్ణాంకం      id;
    శూన్య *   va;
    size_t   కొలత;
    uint64_t phys_addr;
    uint64_t cpu_affinity_ముసుగు;
    పూర్ణాంకం      owner_tid;
    పూర్ణాంకం      is_acquired;
} spine_tcm_ఖండం_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API స్థిరరాశి అక్షరం * spine_tcm_runtime_సంస్కరణ(శూన్య);
SPINE_TCM_API పూర్ణాంకం          spine_tcm_runtime_is_అందుబాటు(శూన్య);
SPINE_TCM_API పూర్ణాంకం          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API పూర్ణాంకం          spine_tcm_runtime_mem_info(పూర్ణాంకం id, spine_tcm_ఖండం_info_t * info);
SPINE_TCM_API శూన్య *       spine_tcm_runtime_mem_get(పూర్ణాంకం id);
SPINE_TCM_API పూర్ణాంకం          spine_tcm_runtime_mem_free(పూర్ణాంకం id);
SPINE_TCM_API శూన్య *       spine_tcm_runtime_mem_try_wait(పూర్ణాంకం id, size_t timeout_us);
SPINE_TCM_API పూర్ణాంకం          spine_tcm_runtime_mem_release(పూర్ణాంకం id);
SPINE_TCM_API పూర్ణాంకం          spine_tcm_runtime_mem_force_release(పూర్ణాంకం id);
SPINE_TCM_API పూర్ణాంకం          spine_tcm_runtime_mem_query(పూర్ణాంకం id);

#ఒకవేళ defined(SPINE_TCM_ప్రత్యక్ష_LINK)
/* Optional no-op in direct-link mode. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_open_handle(స్థిరరాశి అక్షరం * so_మార్గం) {
    (శూన్య) so_మార్గం;
    తిరిగి 0;
}

స్థిర అంతఃవరుస స్థిరరాశి అక్షరం * spine_tcm_సంస్కరణ(శూన్య) {
    తిరిగి spine_tcm_runtime_సంస్కరణ();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_is_అందుబాటు(శూన్య) {
    తిరిగి spine_tcm_runtime_is_అందుబాటు();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    తిరిగి spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_ఖండం_info(పూర్ణాంకం id, spine_tcm_ఖండం_info_t * info) {
    తిరిగి spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
స్థిర అంతఃవరుస శూన్య * spine_tcm_mem_get(పూర్ణాంకం id) {
    తిరిగి spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_free(పూర్ణాంకం id) {
    తిరిగి spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
స్థిర అంతఃవరుస శూన్య * spine_tcm_mem_try_wait(పూర్ణాంకం id, size_t over_సమయం) {
    తిరిగి spine_tcm_runtime_mem_try_wait(id, over_సమయం);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_release(పూర్ణాంకం id) {
    తిరిగి spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_force_release(పూర్ణాంకం id) {
    తిరిగి spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_query(పూర్ణాంకం id) {
    తిరిగి spine_tcm_runtime_mem_query(id);
}
#లేదావేళ !defined(SPINE_TCM_BUILD_SHARED)
రకనిర్వచనం నిర్మాణం spine_tcm_handle {
    శూన్య * module_handle;
    పూర్ణాంకం    use_global_scope;
    పూర్ణాంకం    owns_module_handle;
    స్థిరరాశి అక్షరం * (*runtime_సంస్కరణ)(శూన్య);
    పూర్ణాంకం (*runtime_is_అందుబాటు)(శూన్య);
    పూర్ణాంకం (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    పూర్ణాంకం (*runtime_mem_info)(పూర్ణాంకం id, spine_tcm_ఖండం_info_t * info);
    శూన్య * (*runtime_mem_get)(పూర్ణాంకం id);
    పూర్ణాంకం (*runtime_mem_free)(పూర్ణాంకం id);
    శూన్య * (*runtime_mem_try_wait)(పూర్ణాంకం id, size_t over_సమయం);
    పూర్ణాంకం (*runtime_mem_release)(పూర్ణాంకం id);
    పూర్ణాంకం (*runtime_mem_force_release)(పూర్ణాంకం id);
    పూర్ణాంకం (*runtime_mem_query)(పూర్ణాంకం id);
} spine_tcm_handle_t;

స్థిర అంతఃవరుస spine_tcm_handle_t * spine_tcm_default_handle(శూన్య) {
    స్థిర spine_tcm_handle_t handle = { 0 };
    తిరిగి &handle;
}

స్థిర అంతఃవరుస శూన్య spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    ఒకవేళ (handle != ఖాళీ) {
        memset(handle, 0, పరిమాణం(*handle));
    }
}

స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    శూన్య * చిహ్నం_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_సంస్కరణ      = (స్థిరరాశి అక్షరం * (*) (శూన్య) ) dlsym(చిహ్నం_scope, "spine_tcm_runtime_version");
    handle->runtime_is_అందుబాటు = (పూర్ణాంకం (*)(శూన్య)) dlsym(చిహ్నం_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (పూర్ణాంకం (*)(spine_tcm_mem_info_t *)) dlsym(చిహ్నం_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (పూర్ణాంకం (*)(పూర్ణాంకం, spine_tcm_ఖండం_info_t *)) dlsym(చిహ్నం_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (శూన్య * (*) (పూర్ణాంకం) ) dlsym(చిహ్నం_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (పూర్ణాంకం (*)(పూర్ణాంకం)) dlsym(చిహ్నం_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (శూన్య * (*) (పూర్ణాంకం, size_t)) dlsym(చిహ్నం_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (పూర్ణాంకం (*)(పూర్ణాంకం)) dlsym(చిహ్నం_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (పూర్ణాంకం (*)(పూర్ణాంకం)) dlsym(చిహ్నం_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (పూర్ణాంకం (*)(పూర్ణాంకం)) dlsym(చిహ్నం_scope, "spine_tcm_runtime_mem_query");

    తిరిగి handle->runtime_సంస్కరణ != ఖాళీ && handle->runtime_is_అందుబాటు != ఖాళీ &&
                   handle->runtime_layout_info != ఖాళీ && handle->runtime_mem_info != ఖాళీ &&
                   handle->runtime_mem_get != ఖాళీ && handle->runtime_mem_free != ఖాళీ &&
                   handle->runtime_mem_try_wait != ఖాళీ && handle->runtime_mem_release != ఖాళీ &&
                   handle->runtime_mem_force_release != ఖాళీ && handle->runtime_mem_query != ఖాళీ ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    ఒకవేళ (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == ఖాళీ) {
        తిరిగి -1;
    }

    handle->use_global_scope = 1;
    తిరిగి spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_open_handle(స్థిరరాశి అక్షరం * so_మార్గం) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    స్థిరరాశి అక్షరం *         library  = (so_మార్గం != ఖాళీ && so_మార్గం[0] != '\0') ? so_మార్గం : "libspine_tcm.so";

    ఒకవేళ (resolved->module_handle != ఖాళీ || resolved->use_global_scope) {
        తిరిగి 0;
    }

    ఒకవేళ (spine_tcm_try_bind_global(resolved) == 0) {
        తిరిగి 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != ఖాళీ ? 1 : 0;

    ఒకవేళ (resolved->module_handle == ఖాళీ) {
        spine_tcm_handle_reset(resolved);
        తిరిగి -1;
    }

    ఒకవేళ (spine_tcm_handle_bind(resolved) != 0) {
        ఒకవేళ (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        తిరిగి -1;
    }

    తిరిగి 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_is_అందుబాటు(శూన్య) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ ((resolved->module_handle == ఖాళీ && !resolved->use_global_scope) || resolved->runtime_is_అందుబాటు == ఖాళీ) {
        తిరిగి 0;
    }

    తిరిగి resolved->runtime_is_అందుబాటు();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ ((resolved->module_handle == ఖాళీ && !resolved->use_global_scope) || resolved->runtime_layout_info == ఖాళీ) {
        తిరిగి -1;
    }

    తిరిగి resolved->runtime_layout_info(info);
}

స్థిర అంతఃవరుస స్థిరరాశి అక్షరం * spine_tcm_సంస్కరణ(శూన్య) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ ((resolved->module_handle == ఖాళీ && !resolved->use_global_scope) || resolved->runtime_సంస్కరణ == ఖాళీ) {
        తిరిగి "unknown";
    }

    తిరిగి resolved->runtime_సంస్కరణ();
}

/* Returns per-block runtime metadata for the given TCM id. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_ఖండం_info(పూర్ణాంకం id, spine_tcm_ఖండం_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ ((resolved->module_handle == ఖాళీ && !resolved->use_global_scope) || resolved->runtime_mem_info == ఖాళీ) {
        తిరిగి -1;
    }

    తిరిగి resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
స్థిర అంతఃవరుస శూన్య * spine_tcm_mem_get(పూర్ణాంకం id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (resolved->runtime_mem_get == ఖాళీ) {
        తిరిగి ఖాళీ;
    }

    తిరిగి resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_free(పూర్ణాంకం id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ ((resolved->module_handle == ఖాళీ && !resolved->use_global_scope) || resolved->runtime_mem_free == ఖాళీ) {
        తిరిగి -1;
    }

    తిరిగి resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
స్థిర అంతఃవరుస శూన్య * spine_tcm_mem_try_wait(పూర్ణాంకం id, size_t over_సమయం) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (resolved->runtime_mem_try_wait == ఖాళీ) {
        తిరిగి ఖాళీ;
    }

    తిరిగి resolved->runtime_mem_try_wait(id, over_సమయం);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_release(పూర్ణాంకం id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ ((resolved->module_handle == ఖాళీ && !resolved->use_global_scope) || resolved->runtime_mem_release == ఖాళీ) {
        తిరిగి -1;
    }

    తిరిగి resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_force_release(పూర్ణాంకం id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ ((resolved->module_handle == ఖాళీ && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == ఖాళీ) {
        తిరిగి -1;
    }

    తిరిగి resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
స్థిర అంతఃవరుస పూర్ణాంకం spine_tcm_mem_query(పూర్ణాంకం id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ఒకవేళ (resolved->module_handle == ఖాళీ && !resolved->use_global_scope) {
        (శూన్య) spine_tcm_open_handle(ఖాళీ);
    }

    ఒకవేళ ((resolved->module_handle == ఖాళీ && !resolved->use_global_scope) || resolved->runtime_mem_query == ఖాళీ) {
        తిరిగి -1;
    }

    తిరిగి resolved->runtime_mem_query(id);
}
#లేకపోతే
స్థిర అంతఃవరుస స్థిరరాశి అక్షరం * spine_tcm_సంస్కరణ(శూన్య) {
    తిరిగి spine_tcm_runtime_సంస్కరణ();
}
#అయితే_ముగింపు

#నిర్వచించు SPINE_TCM_సంస్కరణ (spine_tcm_సంస్కరణ())

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు

#అయితే_ముగింపు
