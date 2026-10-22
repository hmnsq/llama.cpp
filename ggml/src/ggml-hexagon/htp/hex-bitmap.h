#నిర్వచితం_కాకపోతే HEX_BITMAP_H
#నిర్వచించు HEX_BITMAP_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <string.h>

స్థిర అంతఃవరుస శూన్య bitmap_సమితి(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

స్థిర అంతఃవరుస శూన్య bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

స్థిర అంతఃవరుస bool bitmap_test(స్థిరరాశి uint32_t * bitmap, uint32_t idx) {
    తిరిగి (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

స్థిర అంతఃవరుస శూన్య bitmap_reset(uint32_t * bitmap, size_t కొలత_లోపల_బిట్లు) {
    memset(bitmap, 0, ((కొలత_లోపల_బిట్లు + 31) / 32) * పరిమాణం(uint32_t));
}

#అయితే_ముగింపు // HEX_BITMAP_H
