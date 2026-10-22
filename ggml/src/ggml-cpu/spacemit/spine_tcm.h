#اگر_غیر_تعریف_شدہ SPINE_TCM_عوامی_H
#تعریف_کرو SPINE_TCM_عوامی_H

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

#اگر !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_براہ_راست_LINK)
#    include <dlfcn.h>
#اگر_ختم

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

#اگر defined(_WIN32)
#    اگر defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    ورنہ
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#ورنہ
#    define SPINE_TCM_API __attribute__((visibility("default")))
#اگر_ختم

قسمتعریف ساخت spine_tcm_mem_info {
    size_t blk_سائز;
    size_t blk_عدد;
    صحیح    is_fake_tcm;
} spine_tcm_mem_info_t;

قسمتعریف ساخت spine_tcm_بلاک_info {
    صحیح      id;
    خالی *   va;
    size_t   سائز;
    uint64_t phys_addr;
    uint64_t cpu_affinity_نقاب;
    صحیح      owner_tid;
    صحیح      is_acquired;
} spine_tcm_بلاک_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ثابت حرف * spine_tcm_runtime_نسخہ(خالی);
SPINE_TCM_API صحیح          spine_tcm_runtime_is_دستیاب(خالی);
SPINE_TCM_API صحیح          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_info(صحیح id, spine_tcm_بلاک_info_t * info);
SPINE_TCM_API خالی *       spine_tcm_runtime_mem_get(صحیح id);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_free(صحیح id);
SPINE_TCM_API خالی *       spine_tcm_runtime_mem_try_wait(صحیح id, size_t timeout_us);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_release(صحیح id);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_force_release(صحیح id);
SPINE_TCM_API صحیح          spine_tcm_runtime_mem_query(صحیح id);

#اگر defined(SPINE_TCM_براہ_راست_LINK)
/* Optional no-op in direct-link mode. */
ساکن اندرونی صحیح spine_tcm_open_handle(ثابت حرف * so_راستہ) {
    (خالی) so_راستہ;
    واپس 0;
}

ساکن اندرونی ثابت حرف * spine_tcm_نسخہ(خالی) {
    واپس spine_tcm_runtime_نسخہ();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ساکن اندرونی صحیح spine_tcm_is_دستیاب(خالی) {
    واپس spine_tcm_runtime_is_دستیاب();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ساکن اندرونی صحیح spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    واپس spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
ساکن اندرونی صحیح spine_tcm_بلاک_info(صحیح id, spine_tcm_بلاک_info_t * info) {
    واپس spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ساکن اندرونی خالی * spine_tcm_mem_get(صحیح id) {
    واپس spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ساکن اندرونی صحیح spine_tcm_mem_free(صحیح id) {
    واپس spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ساکن اندرونی خالی * spine_tcm_mem_try_wait(صحیح id, size_t over_وقت) {
    واپس spine_tcm_runtime_mem_try_wait(id, over_وقت);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ساکن اندرونی صحیح spine_tcm_mem_release(صحیح id) {
    واپس spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ساکن اندرونی صحیح spine_tcm_mem_force_release(صحیح id) {
    واپس spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ساکن اندرونی صحیح spine_tcm_mem_query(صحیح id) {
    واپس spine_tcm_runtime_mem_query(id);
}
#وگرنہ_اگر !defined(SPINE_TCM_BUILD_SHARED)
قسمتعریف ساخت spine_tcm_handle {
    خالی * module_handle;
    صحیح    use_global_scope;
    صحیح    owns_module_handle;
    ثابت حرف * (*runtime_نسخہ)(خالی);
    صحیح (*runtime_is_دستیاب)(خالی);
    صحیح (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    صحیح (*runtime_mem_info)(صحیح id, spine_tcm_بلاک_info_t * info);
    خالی * (*runtime_mem_get)(صحیح id);
    صحیح (*runtime_mem_free)(صحیح id);
    خالی * (*runtime_mem_try_wait)(صحیح id, size_t over_وقت);
    صحیح (*runtime_mem_release)(صحیح id);
    صحیح (*runtime_mem_force_release)(صحیح id);
    صحیح (*runtime_mem_query)(صحیح id);
} spine_tcm_handle_t;

ساکن اندرونی spine_tcm_handle_t * spine_tcm_default_handle(خالی) {
    ساکن spine_tcm_handle_t handle = { 0 };
    واپس &handle;
}

ساکن اندرونی خالی spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    اگر (handle != تہی) {
        memset(handle, 0, حجم(*handle));
    }
}

ساکن اندرونی صحیح spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    خالی * علامت_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_نسخہ      = (ثابت حرف * (*) (خالی) ) dlsym(علامت_scope, "spine_tcm_runtime_version");
    handle->runtime_is_دستیاب = (صحیح (*)(خالی)) dlsym(علامت_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (صحیح (*)(spine_tcm_mem_info_t *)) dlsym(علامت_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (صحیح (*)(صحیح, spine_tcm_بلاک_info_t *)) dlsym(علامت_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (خالی * (*) (صحیح) ) dlsym(علامت_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (صحیح (*)(صحیح)) dlsym(علامت_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (خالی * (*) (صحیح, size_t)) dlsym(علامت_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (صحیح (*)(صحیح)) dlsym(علامت_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (صحیح (*)(صحیح)) dlsym(علامت_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (صحیح (*)(صحیح)) dlsym(علامت_scope, "spine_tcm_runtime_mem_query");

    واپس handle->runtime_نسخہ != تہی && handle->runtime_is_دستیاب != تہی &&
                   handle->runtime_layout_info != تہی && handle->runtime_mem_info != تہی &&
                   handle->runtime_mem_get != تہی && handle->runtime_mem_free != تہی &&
                   handle->runtime_mem_try_wait != تہی && handle->runtime_mem_release != تہی &&
                   handle->runtime_mem_force_release != تہی && handle->runtime_mem_query != تہی ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
ساکن اندرونی صحیح spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    اگر (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == تہی) {
        واپس -1;
    }

    handle->use_global_scope = 1;
    واپس spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
ساکن اندرونی صحیح spine_tcm_open_handle(ثابت حرف * so_راستہ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ثابت حرف *         library  = (so_راستہ != تہی && so_راستہ[0] != '\0') ? so_راستہ : "libspine_tcm.so";

    اگر (resolved->module_handle != تہی || resolved->use_global_scope) {
        واپس 0;
    }

    اگر (spine_tcm_try_bind_global(resolved) == 0) {
        واپس 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != تہی ? 1 : 0;

    اگر (resolved->module_handle == تہی) {
        spine_tcm_handle_reset(resolved);
        واپس -1;
    }

    اگر (spine_tcm_handle_bind(resolved) != 0) {
        اگر (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        واپس -1;
    }

    واپس 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ساکن اندرونی صحیح spine_tcm_is_دستیاب(خالی) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر ((resolved->module_handle == تہی && !resolved->use_global_scope) || resolved->runtime_is_دستیاب == تہی) {
        واپس 0;
    }

    واپس resolved->runtime_is_دستیاب();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ساکن اندرونی صحیح spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر ((resolved->module_handle == تہی && !resolved->use_global_scope) || resolved->runtime_layout_info == تہی) {
        واپس -1;
    }

    واپس resolved->runtime_layout_info(info);
}

ساکن اندرونی ثابت حرف * spine_tcm_نسخہ(خالی) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر ((resolved->module_handle == تہی && !resolved->use_global_scope) || resolved->runtime_نسخہ == تہی) {
        واپس "unknown";
    }

    واپس resolved->runtime_نسخہ();
}

/* Returns per-block runtime metadata for the given TCM id. */
ساکن اندرونی صحیح spine_tcm_بلاک_info(صحیح id, spine_tcm_بلاک_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر ((resolved->module_handle == تہی && !resolved->use_global_scope) || resolved->runtime_mem_info == تہی) {
        واپس -1;
    }

    واپس resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ساکن اندرونی خالی * spine_tcm_mem_get(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        واپس تہی;
    }

    اگر (resolved->runtime_mem_get == تہی) {
        واپس تہی;
    }

    واپس resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ساکن اندرونی صحیح spine_tcm_mem_free(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر ((resolved->module_handle == تہی && !resolved->use_global_scope) || resolved->runtime_mem_free == تہی) {
        واپس -1;
    }

    واپس resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ساکن اندرونی خالی * spine_tcm_mem_try_wait(صحیح id, size_t over_وقت) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        واپس تہی;
    }

    اگر (resolved->runtime_mem_try_wait == تہی) {
        واپس تہی;
    }

    واپس resolved->runtime_mem_try_wait(id, over_وقت);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ساکن اندرونی صحیح spine_tcm_mem_release(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر ((resolved->module_handle == تہی && !resolved->use_global_scope) || resolved->runtime_mem_release == تہی) {
        واپس -1;
    }

    واپس resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ساکن اندرونی صحیح spine_tcm_mem_force_release(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر ((resolved->module_handle == تہی && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == تہی) {
        واپس -1;
    }

    واپس resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ساکن اندرونی صحیح spine_tcm_mem_query(صحیح id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    اگر (resolved->module_handle == تہی && !resolved->use_global_scope) {
        (خالی) spine_tcm_open_handle(تہی);
    }

    اگر ((resolved->module_handle == تہی && !resolved->use_global_scope) || resolved->runtime_mem_query == تہی) {
        واپس -1;
    }

    واپس resolved->runtime_mem_query(id);
}
#ورنہ
ساکن اندرونی ثابت حرف * spine_tcm_نسخہ(خالی) {
    واپس spine_tcm_runtime_نسخہ();
}
#اگر_ختم

#تعریف_کرو SPINE_TCM_نسخہ (spine_tcm_نسخہ())

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم

#اگر_ختم
