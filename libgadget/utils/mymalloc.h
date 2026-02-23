#ifndef _MYMALLOC_H_
#define _MYMALLOC_H_

#include "memory.h"
#include "system.h"

extern Allocator A_MAIN[1];
extern Allocator A_TEMP[1];

/* Initialize the main memory block*/
void mymalloc_init(double MemoryMB);
/* Initialize the small temporary memory block*/
void tamalloc_init(void);
void report_detailed_memory_usage(const char *label, const char * fmt, ...);

/* Memory allocated with this mechanism is tracked by our allocator framework but allocated with posix_memalign */
#define  mymalloc(name, size)            allocator_alloc_bot(A_MAIN, name, size)
#define  mymalloc2(name, size)           allocator_alloc_top(A_MAIN, name, size)

/* Memory allocated with this mechanism is allocated purely on the gpu
 * but tracked with our framework so we can see how much meory we are using. */
#define mycudamalloc(name, size)        allocator_alloc_bot(A_MAIN, name, size)

/* Memory allocated with this mechanism is allocated mapped: primarily on the CPU,
 * but accessible on the gpu as well, copied over on page faults.*/
#define mycudamalloc(name, size)        allocator_alloc_bot(A_MAIN, name, size)

#define  myrealloc(ptr, size)     allocator_realloc(A_MAIN, ptr, size)
#define  myfree(x)                 allocator_free(x)

#define  ma_malloc(name, type, nele)            (type*) allocator_alloc_bot(A_MAIN, name, sizeof(type) * (nele))
#define  ma_malloc2(name, type, nele)           (type*) allocator_alloc_top(A_MAIN, name, sizeof(type) * (nele))
#define  ma_free(p) allocator_free(p)

#define  ta_malloc(name, type, nele)            (type*) allocator_alloc_bot(A_TEMP, name, sizeof(type) * (nele))
#define  ta_malloc2(name, type, nele)           (type*) allocator_alloc_top(A_TEMP, name, sizeof(type) * (nele))
#define  ta_reset()     allocator_reset(A_TEMP, 0)
#define  ta_free(p) allocator_free(p)

#define  report_memory_usage(x)    report_detailed_memory_usage(x, "%s:%d", __FILE__, __LINE__)
#define  mymalloc_freebytes()       get_freemem_bytes()
#define  mymalloc_usedbytes()       allocator_get_used_size(A_MAIN, ALLOC_DIR_BOTH)

#endif
