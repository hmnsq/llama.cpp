#अगर_अघोषित SPINE_TCM_PUBLIC_H_
#घोषणा SPINE_TCM_PUBLIC_H_

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

#समावेश <stddef.h>
#समावेश <stdint.h>
#समावेश <string.h>

#अगर !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#पूर्ण_अगर

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

#अगर defined(_WIN32)
#    अगर defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    अन्यथा
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#अन्यथा
#    define SPINE_TCM_API __attribute__((visibility("default")))
#पूर्ण_अगर

प्रकार काष्ठा spine_tcm_mem_info {
    माप_प्रकार blk_आकार;
    माप_प्रकार blk_अंक;
    पूर्णांक    is_fake_tcm;
} spine_tcm_mem_info_t;

प्रकार काष्ठा spine_tcm_block_info {
    पूर्णांक      id;
    व्योम *   va;
    माप_प्रकार   आकार;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    पूर्णांक      owner_tid;
    पूर्णांक      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API स्थिर अक्षर * spine_tcm_runtime_version(व्योम);
SPINE_TCM_API पूर्णांक          spine_tcm_runtime_is_available(व्योम);
SPINE_TCM_API पूर्णांक          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API पूर्णांक          spine_tcm_runtime_mem_info(पूर्णांक id, spine_tcm_block_info_t * info);
SPINE_TCM_API व्योम *       spine_tcm_runtime_mem_get(पूर्णांक id);
SPINE_TCM_API पूर्णांक          spine_tcm_runtime_mem_free(पूर्णांक id);
SPINE_TCM_API व्योम *       spine_tcm_runtime_mem_try_wait(पूर्णांक id, माप_प्रकार timeout_us);
SPINE_TCM_API पूर्णांक          spine_tcm_runtime_mem_release(पूर्णांक id);
SPINE_TCM_API पूर्णांक          spine_tcm_runtime_mem_force_release(पूर्णांक id);
SPINE_TCM_API पूर्णांक          spine_tcm_runtime_mem_query(पूर्णांक id);

#अगर defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
अटल अंतरभूत पूर्णांक spine_tcm_open_handle(स्थिर अक्षर * so_path) {
    (व्योम) so_path;
    वापस 0;
}

अटल अंतरभूत स्थिर अक्षर * spine_tcm_version(व्योम) {
    वापस spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
अटल अंतरभूत पूर्णांक spine_tcm_is_available(व्योम) {
    वापस spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
अटल अंतरभूत पूर्णांक spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    वापस spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
अटल अंतरभूत पूर्णांक spine_tcm_block_info(पूर्णांक id, spine_tcm_block_info_t * info) {
    वापस spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
अटल अंतरभूत व्योम * spine_tcm_mem_get(पूर्णांक id) {
    वापस spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
अटल अंतरभूत पूर्णांक spine_tcm_mem_free(पूर्णांक id) {
    वापस spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
अटल अंतरभूत व्योम * spine_tcm_mem_try_wait(पूर्णांक id, माप_प्रकार over_time) {
    वापस spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
अटल अंतरभूत पूर्णांक spine_tcm_mem_release(पूर्णांक id) {
    वापस spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
अटल अंतरभूत पूर्णांक spine_tcm_mem_force_release(पूर्णांक id) {
    वापस spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
अटल अंतरभूत पूर्णांक spine_tcm_mem_query(पूर्णांक id) {
    वापस spine_tcm_runtime_mem_query(id);
}
#या_अगर !defined(SPINE_TCM_BUILD_SHARED)
प्रकार काष्ठा spine_tcm_handle {
    व्योम * module_handle;
    पूर्णांक    use_global_scope;
    पूर्णांक    owns_module_handle;
    स्थिर अक्षर * (*runtime_version)(व्योम);
    पूर्णांक (*runtime_is_available)(व्योम);
    पूर्णांक (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    पूर्णांक (*runtime_mem_info)(पूर्णांक id, spine_tcm_block_info_t * info);
    व्योम * (*runtime_mem_get)(पूर्णांक id);
    पूर्णांक (*runtime_mem_free)(पूर्णांक id);
    व्योम * (*runtime_mem_try_wait)(पूर्णांक id, माप_प्रकार over_time);
    पूर्णांक (*runtime_mem_release)(पूर्णांक id);
    पूर्णांक (*runtime_mem_force_release)(पूर्णांक id);
    पूर्णांक (*runtime_mem_query)(पूर्णांक id);
} spine_tcm_handle_t;

अटल अंतरभूत spine_tcm_handle_t * spine_tcm_default_handle(व्योम) {
    अटल spine_tcm_handle_t handle = { 0 };
    वापस &handle;
}

अटल अंतरभूत व्योम spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    अगर (handle != शून्य) {
        स_रखो(handle, 0, माप(*handle));
    }
}

अटल अंतरभूत पूर्णांक spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    व्योम * प्रतीक_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (स्थिर अक्षर * (*) (व्योम) ) dlsym(प्रतीक_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (पूर्णांक (*)(व्योम)) dlsym(प्रतीक_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (पूर्णांक (*)(spine_tcm_mem_info_t *)) dlsym(प्रतीक_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (पूर्णांक (*)(पूर्णांक, spine_tcm_block_info_t *)) dlsym(प्रतीक_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (व्योम * (*) (पूर्णांक) ) dlsym(प्रतीक_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (पूर्णांक (*)(पूर्णांक)) dlsym(प्रतीक_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (व्योम * (*) (पूर्णांक, माप_प्रकार)) dlsym(प्रतीक_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (पूर्णांक (*)(पूर्णांक)) dlsym(प्रतीक_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (पूर्णांक (*)(पूर्णांक)) dlsym(प्रतीक_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (पूर्णांक (*)(पूर्णांक)) dlsym(प्रतीक_scope, "spine_tcm_runtime_mem_query");

    वापस handle->runtime_version != शून्य && handle->runtime_is_available != शून्य &&
                   handle->runtime_layout_info != शून्य && handle->runtime_mem_info != शून्य &&
                   handle->runtime_mem_get != शून्य && handle->runtime_mem_free != शून्य &&
                   handle->runtime_mem_try_wait != शून्य && handle->runtime_mem_release != शून्य &&
                   handle->runtime_mem_force_release != शून्य && handle->runtime_mem_query != शून्य ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
अटल अंतरभूत पूर्णांक spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    अगर (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == शून्य) {
        वापस -1;
    }

    handle->use_global_scope = 1;
    वापस spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
अटल अंतरभूत पूर्णांक spine_tcm_open_handle(स्थिर अक्षर * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    स्थिर अक्षर *         library  = (so_path != शून्य && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    अगर (resolved->module_handle != शून्य || resolved->use_global_scope) {
        वापस 0;
    }

    अगर (spine_tcm_try_bind_global(resolved) == 0) {
        वापस 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != शून्य ? 1 : 0;

    अगर (resolved->module_handle == शून्य) {
        spine_tcm_handle_reset(resolved);
        वापस -1;
    }

    अगर (spine_tcm_handle_bind(resolved) != 0) {
        अगर (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        वापस -1;
    }

    वापस 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
अटल अंतरभूत पूर्णांक spine_tcm_is_available(व्योम) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर ((resolved->module_handle == शून्य && !resolved->use_global_scope) || resolved->runtime_is_available == शून्य) {
        वापस 0;
    }

    वापस resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
अटल अंतरभूत पूर्णांक spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर ((resolved->module_handle == शून्य && !resolved->use_global_scope) || resolved->runtime_layout_info == शून्य) {
        वापस -1;
    }

    वापस resolved->runtime_layout_info(info);
}

अटल अंतरभूत स्थिर अक्षर * spine_tcm_version(व्योम) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर ((resolved->module_handle == शून्य && !resolved->use_global_scope) || resolved->runtime_version == शून्य) {
        वापस "unknown";
    }

    वापस resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
अटल अंतरभूत पूर्णांक spine_tcm_block_info(पूर्णांक id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर ((resolved->module_handle == शून्य && !resolved->use_global_scope) || resolved->runtime_mem_info == शून्य) {
        वापस -1;
    }

    वापस resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
अटल अंतरभूत व्योम * spine_tcm_mem_get(पूर्णांक id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        वापस शून्य;
    }

    अगर (resolved->runtime_mem_get == शून्य) {
        वापस शून्य;
    }

    वापस resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
अटल अंतरभूत पूर्णांक spine_tcm_mem_free(पूर्णांक id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर ((resolved->module_handle == शून्य && !resolved->use_global_scope) || resolved->runtime_mem_free == शून्य) {
        वापस -1;
    }

    वापस resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
अटल अंतरभूत व्योम * spine_tcm_mem_try_wait(पूर्णांक id, माप_प्रकार over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        वापस शून्य;
    }

    अगर (resolved->runtime_mem_try_wait == शून्य) {
        वापस शून्य;
    }

    वापस resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
अटल अंतरभूत पूर्णांक spine_tcm_mem_release(पूर्णांक id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर ((resolved->module_handle == शून्य && !resolved->use_global_scope) || resolved->runtime_mem_release == शून्य) {
        वापस -1;
    }

    वापस resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
अटल अंतरभूत पूर्णांक spine_tcm_mem_force_release(पूर्णांक id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर ((resolved->module_handle == शून्य && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == शून्य) {
        वापस -1;
    }

    वापस resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
अटल अंतरभूत पूर्णांक spine_tcm_mem_query(पूर्णांक id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    अगर (resolved->module_handle == शून्य && !resolved->use_global_scope) {
        (व्योम) spine_tcm_open_handle(शून्य);
    }

    अगर ((resolved->module_handle == शून्य && !resolved->use_global_scope) || resolved->runtime_mem_query == शून्य) {
        वापस -1;
    }

    वापस resolved->runtime_mem_query(id);
}
#अन्यथा
अटल अंतरभूत स्थिर अक्षर * spine_tcm_version(व्योम) {
    वापस spine_tcm_runtime_version();
}
#पूर्ण_अगर

#घोषणा SPINE_TCM_VERSION (spine_tcm_version())

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर

#पूर्ण_अगर
