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

#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#ଯଦି !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#    include <dlfcn.h>
#endif

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

#ଯଦି defined(_WIN32)
#    ଯଦି defined(SPINE_TCM_BUILD_SHARED)
#        define SPINE_TCM_API __declspec(dllexport)
#    ଅନ୍ୟଥା
#        define SPINE_TCM_API __declspec(dllimport)
#    endif
#ଅନ୍ୟଥା
#    define SPINE_TCM_API __attribute__((visibility("default")))
#endif

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା spine_tcm_mem_info {
    size_t blk_size;
    size_t blk_ଅଙ୍କ;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    is_fake_tcm;
} spine_tcm_mem_info_t;

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା spine_tcm_block_info {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      id;
    ଶୂନ୍ୟ *   va;
    size_t   size;
    uint64_t phys_addr;
    uint64_t cpu_affinity_mask;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      owner_tid;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API ଅଚଳ ଅକ୍ଷର * spine_tcm_runtime_version(ଶୂନ୍ୟ);
SPINE_TCM_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା          spine_tcm_runtime_is_available(ଶୂନ୍ୟ);
SPINE_TCM_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା          spine_tcm_runtime_layout_info(spine_tcm_mem_info_t * info);
SPINE_TCM_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା          spine_tcm_runtime_mem_info(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id, spine_tcm_block_info_t * info);
SPINE_TCM_API ଶୂନ୍ୟ *       spine_tcm_runtime_mem_get(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);
SPINE_TCM_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା          spine_tcm_runtime_mem_free(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);
SPINE_TCM_API ଶୂନ୍ୟ *       spine_tcm_runtime_mem_try_wait(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id, size_t timeout_us);
SPINE_TCM_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା          spine_tcm_runtime_mem_release(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);
SPINE_TCM_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା          spine_tcm_runtime_mem_force_release(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);
SPINE_TCM_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା          spine_tcm_runtime_mem_query(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);

#ଯଦି defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_open_handle(ଅଚଳ ଅକ୍ଷର * so_path) {
    (ଶୂନ୍ୟ) so_path;
    ଫେରନ୍ତୁ 0;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ଅକ୍ଷର * spine_tcm_version(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ spine_tcm_runtime_version();
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_is_available(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ spine_tcm_runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    ଫେରନ୍ତୁ spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_block_info(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id, spine_tcm_block_info_t * info) {
    ଫେରନ୍ତୁ spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * spine_tcm_mem_get(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    ଫେରନ୍ତୁ spine_tcm_runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_free(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    ଫେରନ୍ତୁ spine_tcm_runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * spine_tcm_mem_try_wait(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id, size_t over_time) {
    ଫେରନ୍ତୁ spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_release(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    ଫେରନ୍ତୁ spine_tcm_runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_force_release(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    ଫେରନ୍ତୁ spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_query(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    ଫେରନ୍ତୁ spine_tcm_runtime_mem_query(id);
}
#ନଚେତଯଦି !defined(SPINE_TCM_BUILD_SHARED)
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା spine_tcm_handle {
    ଶୂନ୍ୟ * module_handle;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    use_global_scope;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    owns_module_handle;
    ଅଚଳ ଅକ୍ଷର * (*runtime_version)(ଶୂନ୍ୟ);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*runtime_is_available)(ଶୂନ୍ୟ);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*runtime_layout_info)(spine_tcm_mem_info_t * info);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*runtime_mem_info)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id, spine_tcm_block_info_t * info);
    ଶୂନ୍ୟ * (*runtime_mem_get)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*runtime_mem_free)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);
    ଶୂନ୍ୟ * (*runtime_mem_try_wait)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id, size_t over_time);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*runtime_mem_release)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*runtime_mem_force_release)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*runtime_mem_query)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id);
} spine_tcm_handle_t;

ସ୍ଥିର ଧାଡ଼ିଭିତରେ spine_tcm_handle_t * spine_tcm_default_handle(ଶୂନ୍ୟ) {
    ସ୍ଥିର spine_tcm_handle_t handle = { 0 };
    ଫେରନ୍ତୁ &handle;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ spine_tcm_handle_reset(spine_tcm_handle_t * handle) {
    ଯଦି (handle != ଖାଲି) {
        memset(handle, 0, ଆକାର(*handle));
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_handle_bind(spine_tcm_handle_t * handle) {
    ଶୂନ୍ୟ * ପ୍ରତୀକ_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

    handle->runtime_version      = (ଅଚଳ ଅକ୍ଷର * (*) (ଶୂନ୍ୟ) ) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_version");
    handle->runtime_is_available = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*)(ଶୂନ୍ୟ)) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_is_available");
    handle->runtime_layout_info =
        (ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*)(spine_tcm_mem_info_t *)) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_layout_info");
    handle->runtime_mem_info =
        (ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା, spine_tcm_block_info_t *)) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_mem_info");
    handle->runtime_mem_get      = (ଶୂନ୍ୟ * (*) (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_mem_get");
    handle->runtime_mem_free     = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_mem_free");
    handle->runtime_mem_try_wait = (ଶୂନ୍ୟ * (*) (ପୂର୍ଣ୍ଣସଂଖ୍ୟା, size_t)) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_mem_try_wait");
    handle->runtime_mem_release  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_mem_release");
    handle->runtime_mem_force_release = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_mem_force_release");
    handle->runtime_mem_query         = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)) dlsym(ପ୍ରତୀକ_scope, "spine_tcm_runtime_mem_query");

    ଫେରନ୍ତୁ handle->runtime_version != ଖାଲି && handle->runtime_is_available != ଖାଲି &&
                   handle->runtime_layout_info != ଖାଲି && handle->runtime_mem_info != ଖାଲି &&
                   handle->runtime_mem_get != ଖାଲି && handle->runtime_mem_free != ଖାଲି &&
                   handle->runtime_mem_try_wait != ଖାଲି && handle->runtime_mem_release != ଖାଲି &&
                   handle->runtime_mem_force_release != ଖାଲି && handle->runtime_mem_query != ଖାଲି ?
               0 :
               -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_try_bind_global(spine_tcm_handle_t * handle) {
    ଯଦି (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == ଖାଲି) {
        ଫେରନ୍ତୁ -1;
    }

    handle->use_global_scope = 1;
    ଫେରନ୍ତୁ spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_open_handle(ଅଚଳ ଅକ୍ଷର * so_path) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();
    ଅଚଳ ଅକ୍ଷର *         library  = (so_path != ଖାଲି && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

    ଯଦି (resolved->module_handle != ଖାଲି || resolved->use_global_scope) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (spine_tcm_try_bind_global(resolved) == 0) {
        ଫେରନ୍ତୁ 0;
    }

    spine_tcm_handle_reset(resolved);

    resolved->module_handle      = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
    resolved->owns_module_handle = resolved->module_handle != ଖାଲି ? 1 : 0;

    ଯଦି (resolved->module_handle == ଖାଲି) {
        spine_tcm_handle_reset(resolved);
        ଫେରନ୍ତୁ -1;
    }

    ଯଦି (spine_tcm_handle_bind(resolved) != 0) {
        ଯଦି (resolved->owns_module_handle) {
            dlclose(resolved->module_handle);
        }
        spine_tcm_handle_reset(resolved);
        ଫେରନ୍ତୁ -1;
    }

    ଫେରନ୍ତୁ 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_is_available(ଶୂନ୍ୟ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି ((resolved->module_handle == ଖାଲି && !resolved->use_global_scope) || resolved->runtime_is_available == ଖାଲି) {
        ଫେରନ୍ତୁ 0;
    }

    ଫେରନ୍ତୁ resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_info(spine_tcm_mem_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି ((resolved->module_handle == ଖାଲି && !resolved->use_global_scope) || resolved->runtime_layout_info == ଖାଲି) {
        ଫେରନ୍ତୁ -1;
    }

    ଫେରନ୍ତୁ resolved->runtime_layout_info(info);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ଅକ୍ଷର * spine_tcm_version(ଶୂନ୍ୟ) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି ((resolved->module_handle == ଖାଲି && !resolved->use_global_scope) || resolved->runtime_version == ଖାଲି) {
        ଫେରନ୍ତୁ "unknown";
    }

    ଫେରନ୍ତୁ resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_block_info(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id, spine_tcm_block_info_t * info) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି ((resolved->module_handle == ଖାଲି && !resolved->use_global_scope) || resolved->runtime_mem_info == ଖାଲି) {
        ଫେରନ୍ତୁ -1;
    }

    ଫେରନ୍ତୁ resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * spine_tcm_mem_get(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଯଦି (resolved->runtime_mem_get == ଖାଲି) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଫେରନ୍ତୁ resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_free(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି ((resolved->module_handle == ଖାଲି && !resolved->use_global_scope) || resolved->runtime_mem_free == ଖାଲି) {
        ଫେରନ୍ତୁ -1;
    }

    ଫେରନ୍ତୁ resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * spine_tcm_mem_try_wait(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id, size_t over_time) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଯଦି (resolved->runtime_mem_try_wait == ଖାଲି) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଫେରନ୍ତୁ resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_release(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି ((resolved->module_handle == ଖାଲି && !resolved->use_global_scope) || resolved->runtime_mem_release == ଖାଲି) {
        ଫେରନ୍ତୁ -1;
    }

    ଫେରନ୍ତୁ resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_force_release(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି ((resolved->module_handle == ଖାଲି && !resolved->use_global_scope) ||
        resolved->runtime_mem_force_release == ଖାଲି) {
        ଫେରନ୍ତୁ -1;
    }

    ଫେରନ୍ତୁ resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_tcm_mem_query(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    spine_tcm_handle_t * resolved = spine_tcm_default_handle();

    ଯଦି (resolved->module_handle == ଖାଲି && !resolved->use_global_scope) {
        (ଶୂନ୍ୟ) spine_tcm_open_handle(ଖାଲି);
    }

    ଯଦି ((resolved->module_handle == ଖାଲି && !resolved->use_global_scope) || resolved->runtime_mem_query == ଖାଲି) {
        ଫେରନ୍ତୁ -1;
    }

    ଫେରନ୍ତୁ resolved->runtime_mem_query(id);
}
#ଅନ୍ୟଥା
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ଅକ୍ଷର * spine_tcm_version(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ spine_tcm_runtime_version();
}
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
