// File: slots_gc_sorted.cu

// Include necessary headers
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <cub/cub.cuh>

// Define the number of particle types
#define NUM_PARTICLE_TYPES 6

// Define the particle data structure
struct particle_data {
    float Pos[3];
    uint8_t Type;
    uint8_t IsGarbage;
};

// Define the Peano key function (dummy implementation)
__host__ __device__ uint64_t PEANO(float Pos[3], float BoxSize) {
    // Simple hash function for testing purposes
    return (uint64_t)(Pos[0] + Pos[1] + Pos[2]);
}

// Define the particle manager type
struct part_manager_type {
    struct particle_data *Base;
    int64_t NumPart;
    float BoxSize;
};

// Define slot manager info
struct slot_info {
    void *ptr;
    size_t size;
    size_t elsize;
};

// Define the slots manager type
struct slots_manager_type {
    struct slot_info info[NUM_PARTICLE_TYPES];
};

// Placeholder for SLOTS_ENABLED macro
#define SLOTS_ENABLED(ptype, sman) (1)

// Placeholder for qsort_openmp (using standard qsort for simplicity)
#define qsort_openmp(base, num, size, cmp) qsort(base, num, size, cmp)

// Placeholder functions
void *mymalloc(const char *name, size_t size);
void myfree(void *ptr);
void slots_gc_mark(struct part_manager_type *pman, struct slots_manager_type *sman);
size_t slots_get_last_garbage(int start, int end, int ptype,
                              struct part_manager_type *pman,
                              struct slots_manager_type *sman);
void slots_gc_collect(int ptype, struct part_manager_type *pman,
                      struct slots_manager_type *sman);
int slot_cmp_reverse_link(const void *a, const void *b);
void slots_check_id_consistency(struct part_manager_type *pman,
                                struct slots_manager_type *sman);

// Function implementations
void *mymalloc(const char *name, size_t size) {
    return malloc(size);
}

void myfree(void *ptr) {
    free(ptr);
}

void slots_gc_mark(struct part_manager_type *pman, struct slots_manager_type *sman) {
    // Dummy implementation
}

size_t slots_get_last_garbage(int start, int end, int ptype,
                              struct part_manager_type *pman,
                              struct slots_manager_type *sman) {
    // Dummy implementation
    return sman->info[ptype].size;
}

void slots_gc_collect(int ptype, struct part_manager_type *pman,
                      struct slots_manager_type *sman) {
    // Dummy implementation
}

int slot_cmp_reverse_link(const void *a, const void *b) {
    // Dummy comparison function
    return 0;
}

void slots_check_id_consistency(struct part_manager_type *pman,
                                struct slots_manager_type *sman) {
    // Dummy implementation
}

// Corrected slots_gc_sorted function using cub::DeviceRadixSort with uint64_t keys
void slots_gc_sorted(struct part_manager_type *pman, struct slots_manager_type *sman) {
    int ptype;
    int64_t i;
    int64_t garbage = 0;

    // Allocate host arrays
    uint64_t *h_keys = (uint64_t *)malloc(pman->NumPart * sizeof(uint64_t));
    uint64_t *h_values = (uint64_t *)malloc(pman->NumPart * sizeof(uint64_t));

    // Initialize keys and values
    for (i = 0; i < pman->NumPart; i++) {
        uint8_t TypeKey = pman->Base[i].Type;
        uint64_t Key = PEANO(pman->Base[i].Pos, pman->BoxSize);

        // Limit Key to 56 bits
        Key &= 0x00FFFFFFFFFFFFFFULL;  // Mask to keep lower 56 bits

        if (pman->Base[i].IsGarbage) {
            garbage++;
            TypeKey = 255;  // Move garbage to the end
        }

        // Combine TypeKey and Key into a single uint64_t
        h_keys[i] = ((uint64_t)TypeKey << 56) | Key;

        h_values[i] = i;  // Pindex
    }

    // Allocate device arrays
    uint64_t *d_keys_in, *d_keys_out;
    uint64_t *d_values_in, *d_values_out;

    cudaMalloc((void**)&d_keys_in, pman->NumPart * sizeof(uint64_t));
    cudaMalloc((void**)&d_keys_out, pman->NumPart * sizeof(uint64_t));
    
    cudaMalloc((void**)&d_values_in, pman->NumPart * sizeof(uint64_t));
 
    cudaMalloc((void**)&d_values_out, pman->NumPart * sizeof(uint64_t));

    // Copy data to device
    cudaMemcpy(d_keys_in, h_keys, pman->NumPart * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_values_in, h_values, pman->NumPart * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_keys_in, d_keys_out,
                                    d_values_in, d_values_out, pman->NumPart);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Sort the keys and values
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_keys_in, d_keys_out,
                                    d_values_in, d_values_out, pman->NumPart);

    // Copy sorted indices back to host
    cudaMemcpy(h_values, d_values_out, pman->NumPart * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Rearrange pman->Base according to sorted indices
    struct particle_data *temp_p = (struct particle_data *)mymalloc("temp_p", pman->NumPart * sizeof(struct particle_data));
    for (i = 0; i < pman->NumPart; i++) {
        temp_p[i] = pman->Base[h_values[i]];
    }
    memcpy(pman->Base, temp_p, pman->NumPart * sizeof(struct particle_data));
    myfree(temp_p);

    // Remove garbage particles
    pman->NumPart -= garbage;

    // Free device memory
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_temp_storage);

    // Free host memory
    free(h_keys);
    free(h_values);

    // Set up ReverseLink
    slots_gc_mark(pman, sman);

    for (ptype = 0; ptype < NUM_PARTICLE_TYPES; ptype++) {
        if (!SLOTS_ENABLED(ptype, sman))
            continue;
        // Sort the used ones by their location in the P array
        qsort_openmp(sman->info[ptype].ptr,
                     sman->info[ptype].size,
                     sman->info[ptype].elsize,
                     slot_cmp_reverse_link);

        // Reduce slots used
        sman->info[ptype].size = slots_get_last_garbage(0, sman->info[ptype].size - 1, ptype, pman, sman);
        slots_gc_collect(ptype, pman, sman);
    }
#ifdef DEBUG
    slots_check_id_consistency(pman, sman);
#endif
}

// Main function to test the slots_gc_sorted function
int main() {
    // Initialize particle manager
    struct part_manager_type pman;
    pman.NumPart = 1000000;  // Number of particles
    pman.BoxSize = 100.0f;   // Simulation box size

    // Allocate memory for particles
    pman.Base = (struct particle_data *)mymalloc("particles", pman.NumPart * sizeof(struct particle_data));

    // Initialize particles with random data
    for (int64_t i = 0; i < pman.NumPart; i++) {
        pman.Base[i].Pos[0] = (float)rand() / RAND_MAX * pman.BoxSize;
        pman.Base[i].Pos[1] = (float)rand() / RAND_MAX * pman.BoxSize;
        pman.Base[i].Pos[2] = (float)rand() / RAND_MAX * pman.BoxSize;
        pman.Base[i].Type = rand() % NUM_PARTICLE_TYPES;
        pman.Base[i].IsGarbage = (rand() % 100) < 5 ? 1 : 0;  // 5% garbage
    }

    // Initialize slots manager
    struct slots_manager_type sman;
    for (int ptype = 0; ptype < NUM_PARTICLE_TYPES; ptype++) {
        sman.info[ptype].ptr = NULL;
        sman.info[ptype].size = 0;
        sman.info[ptype].elsize = sizeof(int);  // Dummy size
    }

    // Call the slots_gc_sorted function
    slots_gc_sorted(&pman, &sman);

    // Output the number of particles after garbage collection
    printf("Number of particles after garbage collection: %ld\n", pman.NumPart);

    // Clean up
    myfree(pman.Base);

    return 0;
}
