/* SPDX-License-Identifier: MIT */
/*
 * librocket — thin C wrapper over the Rockchip "rocket" NPU DRM-accel uAPI.
 *
 * This is pure syscall-surface plumbing. No NPU/tensor semantics, no register
 * commands. Every function returns 0 on success or -errno on failure (except
 * rocket_mmap_bo which returns the mmap'd pointer or MAP_FAILED).
 */
#ifndef LIBROCKET_H
#define LIBROCKET_H

#include "rocket_accel.h"
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * rocket_open - open a rocket NPU accel device
 * @path: device node path (e.g. "/dev/accel/accel0")
 *
 * Returns the file descriptor on success, or -errno on failure.
 */
int rocket_open(const char *path);

/**
 * rocket_create_bo - allocate a buffer object
 * @fd:               rocket device fd
 * @size:             desired BO size in bytes
 * @handle_out:       on success, written with the GEM handle
 * @dma_address_out:  on success, written with the NPU DMA address
 * @mmap_offset_out:  on success, written with the mmap offset
 *
 * Returns 0 on success, -errno on failure.
 */
int rocket_create_bo(int fd, uint32_t size, uint32_t *handle_out,
                     uint64_t *dma_address_out, uint64_t *mmap_offset_out);

/**
 * rocket_mmap_bo - mmap a BO into userspace
 * @fd:           rocket device fd
 * @mmap_offset:  offset returned by rocket_create_bo
 * @size:         number of bytes to map
 *
 * Returns the mapped pointer, or MAP_FAILED on failure.
 */
void *rocket_mmap_bo(int fd, uint64_t mmap_offset, uint32_t size);

/* rocket_munmap_bo - undo rocket_mmap_bo (the VMA pins the GEM object; GEM_CLOSE
 * alone does NOT free it, so every mmap must be paired with a munmap). */
void rocket_munmap_bo(void *map, uint32_t size);

/**
 * rocket_prep_bo - take CPU ownership of a BO (wait for NPU, sync caches)
 * @fd:         rocket device fd
 * @handle:     GEM handle of the BO
 * @timeout_ns: ABSOLUTE CLOCK_MONOTONIC deadline (ns); INT64_MAX = block until signalled
 *
 * Returns 0 on success, -errno on failure.
 */
int rocket_prep_bo(int fd, uint32_t handle, int64_t timeout_ns);

/**
 * rocket_fini_bo - hand a BO back to the device (sync caches)
 * @fd:     rocket device fd
 * @handle: GEM handle of the BO
 *
 * Returns 0 on success, -errno on failure.
 */
int rocket_fini_bo(int fd, uint32_t handle);

/**
 * rocket_submit - submit NPU jobs
 * @fd:        rocket device fd
 * @jobs:      pointer to an array of struct drm_rocket_job
 * @job_count: number of jobs in the array
 *
 * Returns 0 on success, -errno on failure.
 */
int rocket_submit(int fd, struct drm_rocket_job *jobs, uint32_t job_count);

/* Release a BO handle (DRM_IOCTL_GEM_CLOSE). Returns 0 or -errno. */
int rocket_close_bo(int fd, uint32_t handle);

#ifdef __cplusplus
}
#endif

#endif /* LIBROCKET_H */
