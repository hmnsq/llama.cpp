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

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

#जइ !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
बाह्य "C" {
#endif

#जइ defined(_WIN32)
#    जइ defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    अण्णहा
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#अण्णहा
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

प्रकारनिर्देश संरचना spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_अङ्कः;
    पूर्णाङ्क    is_fake_tcm;
} spine_tcm_mem_info_t;

प्रकारनिर्देश संरचना spine_tcm_block_info {
    पूर्णाङ्क      id;
    शून्य *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    पूर्णाङ्क      owner_tid;
    पूर्णाङ्क      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ध्रुव वर्ण * spine_tcm_runtime_version(शून्य);
SPINE_TCM_API पूर्णाङ्क          spine_tcm_runtime_is_available(शून्य);
SPINE_TCM_API पूर्णाङ्क          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API पूर्णाङ्क          spine_tcm_runtime_mem_info(पूर्णाङ्क id, spine_tcm_block_info_t * info);
SPINE_TCM_API शून्य *       spine_tcm_runtime_mem_get(पूर्णाङ्क id);
SPINE_TCM_API पूर्णाङ्क          spine_tcm_runtime_mem_free(पूर्णाङ्क id);
SPINE_TCM_API शून्य *       spine_tcm_runtime_mem_try_wait(पूर्णाङ्क id, size_t timeout_us);
SPINE_TCM_API पूर्णाङ्क          spine_tcm_runtime_mem_release(पूर्णाङ्क id);
SPINE_TCM_API पूर्णाङ्क          spine_tcm_runtime_mem_force_release(पूर्णाङ्क id);
SPINE_TCM_API पूर्णाङ्क          spine_tcm_runtime_mem_query(पूर्णाङ्क id);

#जइ defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_open_handle(ध्रुव वर्ण * so_path) {
    (शून्य) so_path;
    प्रत्यागम 0;
}

स्थिर अन्तःस्थ ध्रुव वर्ण * spine_tcm_version(शून्य) {
    प्रत्यागम spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_is_available(शून्य) {
    प्रत्यागम spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    प्रत्यागम spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_block_info(पूर्णाङ्क id, spine_tcm_block_info_t * info) {
    प्रत्यागम spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
स्थिर अन्तःस्थ शून्य * spine_tcm_mem_get(पूर्णाङ्क id) {
    प्रत्यागम spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_free(पूर्णाङ्क id) {
    प्रत्यागम spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
स्थिर अन्तःस्थ शून्य * spine_tcm_mem_try_wait(पूर्णाङ्क id, size_t over_time) {
    प्रत्यागम spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_release(पूर्णाङ्क id) {
    प्रत्यागम spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_force_release(पूर्णाङ्क id) {
    प्रत्यागम spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_query(पूर्णाङ्क id) {
    प्रत्यागम spine_tcm_runtime_mem_query(id);
}
#नोचेद्यदि !defined(SPINE_TCM_BUILD_SHARED)
प्रकारनिर्देश संरचना spine_tcm_handle {
    शून्य * module_handle;
    पूर्णाङ्क    use_global_scope;
    पूर्णाङ्क    owns_module_handle;
    ध्रुव वर्ण * (*runtime_version)(शून्य);
    पूर्णाङ्क (*runtime_is_available)(शून्य);
    पूर्णाङ्क (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    पूर्णाङ्क (*runtime_mem_info)(पूर्णाङ्क id, spine_tcm_block_info_t * info);
    शून्य * (*runtime_mem_get)(पूर्णाङ्क id);
    पूर्णाङ्क (*runtime_mem_free)(पूर्णाङ्क id);
    शून्य * (*runtime_mem_try_wait)(पूर्णाङ्क id, size_t over_time);
    पूर्णाङ्क (*runtime_mem_release)(पूर्णाङ्क id);
    पूर्णाङ्क (*runtime_mem_force_release)(पूर्णाङ्क id);
    पूर्णाङ्क (*runtime_mem_query)(पूर्णाङ्क id);
} spine_tcm_handle_t;

स्थिर अन्तःस्थ spine_tcm_handle_t * spine_tcm_default_handle(शून्य) {
    स्थिर spine_tcm_handle_t handle = { 0 };
    प्रत्यागम &handle;
}

स्थिर अन्तःस्थ शून्य spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    जइ (handle != रिक्त) {
        memset(handle, 0, परिमाण(*handle));
    }
}

स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    शून्य * चिह्नम्_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (ध्रुव वर्ण * (*) (शून्य) ) dlsym(चिह्नम्_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (पूर्णाङ्क (*)(शून्य)) dlsym(चिह्नम्_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (पूर्णाङ्क (*)(spine_tcm_mem_info_t *)) dlsym(चिह्नम्_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (पूर्णाङ्क (*)(पूर्णाङ्क, spine_tcm_block_info_t *)) dlsym(चिह्नम्_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (शून्य * (*) (पूर्णाङ्क) ) dlsym(चिह्नम्_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (पूर्णाङ्क (*)(पूर्णाङ्क)) dlsym(चिह्नम्_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (शून्य * (*) (पूर्णाङ्क, size_t)) dlsym(चिह्नम्_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (पूर्णाङ्क (*)(पूर्णाङ्क)) dlsym(चिह्नम्_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (पूर्णाङ्क (*)(पूर्णाङ्क)) dlsym(चिह्नम्_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (पूर्णाङ्क (*)(पूर्णाङ्क)) dlsym(चिह्नम्_scope, "spine_tcm_runtime_mem_query");

    प्रत्यागम handle->runtime_version != रिक्त && handle->runtime_is_available != रिक्त &&
                   handle->runtime_layout_info != रिक्त && handle->runtime_mem_info != रिक्त &&
                   handle->runtime_mem_get != रिक्त && handle->runtime_mem_free != रिक्त &&
                   handle->runtime_mem_try_wait != रिक्त && handle->runtime_mem_release != रिक्त &&
                   handle->runtime_mem_force_release != रिक्त && handle->runtime_mem_query != रिक्त ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    जइ (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == रिक्त) {
        प्रत्यागम -1;
    }

    handle->use_global_scope = 1;
    प्रत्यागम spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_open_handle(ध्रुव वर्ण * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ध्रुव वर्ण *         library  = (so_path != रिक्त && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    जइ (resolved->module_handle != रिक्त || resolved->use_global_scope) {
        प्रत्यागम 0;
    }

    जइ (spine_tcm_try_bind_global(resolved) == 0) {
        प्रत्यागम 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != रिक्त ? 1 : 0;

    जइ (resolved->module_handle == रिक्त) {
        spine_tcm_handle_reset(resolved);
        प्रत्यागम -1;
    }

    जइ (spine_tcm_handle_bind(resolved) != 0) {
        जइ (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        प्रत्यागम -1;
    }

    प्रत्यागम 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_is_available(शून्य) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ ((resolved->module_handle == रिक्त && !resolved->use_global_scope) || resolved->runtime_is_available == रिक्त) {
        प्रत्यागम 0;
    }

    प्रत्यागम resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ ((resolved->module_handle == रिक्त && !resolved->use_global_scope) || resolved->runtime_layout_info == रिक्त) {
        प्रत्यागम -1;
    }

    प्रत्यागम resolved->runtime_layout_info(info);
}

स्थिर अन्तःस्थ ध्रुव वर्ण * spine_tcm_version(शून्य) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ ((resolved->module_handle == रिक्त && !resolved->use_global_scope) || resolved->runtime_version == रिक्त) {
        प्रत्यागम "unknown";
    }

    प्रत्यागम resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_block_info(पूर्णाङ्क id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ ((resolved->module_handle == रिक्त && !resolved->use_global_scope) || resolved->runtime_mem_info == रिक्त) {
        प्रत्यागम -1;
    }

    प्रत्यागम resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
स्थिर अन्तःस्थ शून्य * spine_tcm_mem_get(पूर्णाङ्क id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        प्रत्यागम रिक्त;
    }

    जइ (resolved->runtime_mem_get == रिक्त) {
        प्रत्यागम रिक्त;
    }

    प्रत्यागम resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_free(पूर्णाङ्क id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ ((resolved->module_handle == रिक्त && !resolved->use_global_scope) || resolved->runtime_mem_free == रिक्त) {
        प्रत्यागम -1;
    }

    प्रत्यागम resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
स्थिर अन्तःस्थ शून्य * spine_tcm_mem_try_wait(पूर्णाङ्क id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        प्रत्यागम रिक्त;
    }

    जइ (resolved->runtime_mem_try_wait == रिक्त) {
        प्रत्यागम रिक्त;
    }

    प्रत्यागम resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_release(पूर्णाङ्क id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ ((resolved->module_handle == रिक्त && !resolved->use_global_scope) || resolved->runtime_mem_release == रिक्त) {
        प्रत्यागम -1;
    }

    प्रत्यागम resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_force_release(पूर्णाङ्क id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ ((resolved->module_handle == रिक्त && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == रिक्त) {
        प्रत्यागम -1;
    }

    प्रत्यागम resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
स्थिर अन्तःस्थ पूर्णाङ्क spine_tcm_mem_query(पूर्णाङ्क id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    जइ (resolved->module_handle == रिक्त && !resolved->use_global_scope) {
        (शून्य) spine_tcm_open_handle(रिक्त);
    }

    जइ ((resolved->module_handle == रिक्त && !resolved->use_global_scope) || resolved->runtime_mem_query == रिक्त) {
        प्रत्यागम -1;
    }

    प्रत्यागम resolved->runtime_mem_query(id);
}
#अण्णहा
स्थिर अन्तःस्थ ध्रुव वर्ण * spine_tcm_version(शून्य) {
    प्रत्यागम spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
