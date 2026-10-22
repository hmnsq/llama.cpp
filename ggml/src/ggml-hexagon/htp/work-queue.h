#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_WORK_ಸರತಿ_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_WORK_ಸರತಿ_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*work_ಸರತಿ_func_t)(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ *);

ರಚನೆ work_ಸರತಿ_s;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ work_ಸರತಿ_s * work_ಸರತಿ_t;

#ವ್ಯಾಖ್ಯಾನಿಸು WORK_ಸರತಿ_MAX_N_ಎಳೆಗಳು      10

size_t       work_ಸರತಿ_sizeof(uint32_t n_ಎಳೆಗಳು, uint32_t ಸಾಮರ್ಥ್ಯ, uint32_t ರಾಶಿ_ಗಾತ್ರ);
size_t       work_ಸರತಿ_alignof(ಶೂನ್ಯ);
work_ಸರತಿ_t work_ಸರತಿ_init(ಶೂನ್ಯ * ptr, uint32_t n_ಎಳೆಗಳು, uint32_t ಸಾಮರ್ಥ್ಯ, uint32_t ರಾಶಿ_ಗಾತ್ರ);
ಶೂನ್ಯ         work_ಸರತಿ_free(work_ಸರತಿ_t q);

ಶೂನ್ಯ work_ಸರತಿ_wakeup(work_ಸರತಿ_t q);
ಶೂನ್ಯ work_ಸರತಿ_suspend(work_ಸರತಿ_t q);

bool work_ಸರತಿ_ಸತತ_async(work_ಸರತಿ_t q, work_ಸರತಿ_func_t func, ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n);

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool work_ಸರತಿ_ಸತತ(work_ಸರತಿ_t q, work_ಸರತಿ_func_t func, ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n) {
    ಆದರೆ (n <= 1) {
        func(n, 0, ದತ್ತಾಂಶ);
        ಹಿಂತಿರುಗಿ true;
    }
    ಹಿಂತಿರುಗಿ work_ಸರತಿ_ಸತತ_async(q, func, ದತ್ತಾಂಶ, n);
}

// Legacy compatibility
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ work_ಸರತಿ_func_t     ಕೆಲಸಗಾರ_callback_t;
#ವ್ಯಾಖ್ಯಾನಿಸು ಕೆಲಸಗಾರ_pool_ಸತತ_func  work_ಸರತಿ_ಸತತ
#ವ್ಯಾಖ್ಯಾನಿಸು ಕೆಲಸಗಾರ_pool           work_ಸರತಿ

#ಆದರೆ_ಅಂತ್ಯ  // #ifndef HTP_WORK_QUEUE_H
