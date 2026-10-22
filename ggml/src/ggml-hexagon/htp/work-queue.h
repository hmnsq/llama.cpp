#నిర్వచితం_కాకపోతే HTP_WORK_వరుస_H
#నిర్వచించు HTP_WORK_వరుస_H

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>

రకనిర్వచనం శూన్య (*work_వరుస_func_t)(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య *);

నిర్మాణం work_వరుస_s;
రకనిర్వచనం నిర్మాణం work_వరుస_s * work_వరుస_t;

#నిర్వచించు WORK_వరుస_MAX_N_దారాలు      10

size_t       work_వరుస_sizeof(uint32_t n_దారాలు, uint32_t సామర్థ్యం, uint32_t దొంతర_కొలత);
size_t       work_వరుస_alignof(శూన్య);
work_వరుస_t work_వరుస_init(శూన్య * ptr, uint32_t n_దారాలు, uint32_t సామర్థ్యం, uint32_t దొంతర_కొలత);
శూన్య         work_వరుస_free(work_వరుస_t q);

శూన్య work_వరుస_wakeup(work_వరుస_t q);
శూన్య work_వరుస_suspend(work_వరుస_t q);

bool work_వరుస_నిరంతరం_async(work_వరుస_t q, work_వరుస_func_t func, శూన్య * దత్తాంశం, గుర్తులేని పూర్ణాంకం n);

స్థిర అంతఃవరుస bool work_వరుస_నిరంతరం(work_వరుస_t q, work_వరుస_func_t func, శూన్య * దత్తాంశం, గుర్తులేని పూర్ణాంకం n) {
    ఒకవేళ (n <= 1) {
        func(n, 0, దత్తాంశం);
        తిరిగి true;
    }
    తిరిగి work_వరుస_నిరంతరం_async(q, func, దత్తాంశం, n);
}

// Legacy compatibility
రకనిర్వచనం work_వరుస_func_t     కార్మికుడు_callback_t;
#నిర్వచించు కార్మికుడు_pool_నిరంతరం_func  work_వరుస_నిరంతరం
#నిర్వచించు కార్మికుడు_pool           work_వరుస

#అయితే_ముగింపు  // #ifndef HTP_WORK_QUEUE_H
