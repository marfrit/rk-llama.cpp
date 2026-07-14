/* SPDX-License-Identifier: MIT */
/*
 * librocket — thin C wrapper over the Rockchip "rocket" NPU DRM-accel uAPI.
 *
 * Implementation — pure syscall-surface plumbing.
 */
#include "librocket.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

int rocket_open(const char *path)
{
	int fd = open(path, O_RDWR | O_CLOEXEC);
	if (fd < 0)
		return -errno;
	return fd;
}

int rocket_create_bo(int fd, uint32_t size, uint32_t *handle_out,
		     uint64_t *dma_address_out, uint64_t *mmap_offset_out)
{
	struct drm_rocket_create_bo create_bo;
	int ret;

	memset(&create_bo, 0, sizeof(create_bo));
	create_bo.size = size;

	ret = ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, &create_bo);
	if (ret < 0)
		return -errno;

	if (handle_out)
		*handle_out = create_bo.handle;
	if (dma_address_out)
		*dma_address_out = create_bo.dma_address;
	if (mmap_offset_out)
		*mmap_offset_out = create_bo.offset;

	return 0;
}

void *rocket_mmap_bo(int fd, uint64_t mmap_offset, uint32_t size)
{
	return mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
		    mmap_offset);
}

int rocket_prep_bo(int fd, uint32_t handle, int64_t timeout_ns)
{
	struct drm_rocket_prep_bo prep_bo;
	int ret;

	memset(&prep_bo, 0, sizeof(prep_bo));
	prep_bo.handle = handle;
	prep_bo.timeout_ns = timeout_ns;

	ret = ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &prep_bo);
	if (ret < 0)
		return -errno;
	return 0;
}

int rocket_fini_bo(int fd, uint32_t handle)
{
	struct drm_rocket_fini_bo fini_bo;
	int ret;

	memset(&fini_bo, 0, sizeof(fini_bo));
	fini_bo.handle = handle;

	ret = ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini_bo);
	if (ret < 0)
		return -errno;
	return 0;
}

int rocket_submit(int fd, struct drm_rocket_job *jobs, uint32_t job_count)
{
	struct drm_rocket_submit submit;
	int ret;

	memset(&submit, 0, sizeof(submit));
	submit.jobs = (uintptr_t)jobs;
	submit.job_count = job_count;
	submit.job_struct_size = sizeof(struct drm_rocket_job);

	ret = ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &submit);
	if (ret < 0)
		return -errno;
	return 0;
}

int rocket_close_bo(int fd, uint32_t handle)
{
	struct drm_gem_close c;
	memset(&c, 0, sizeof c);
	c.handle = handle;
	if (ioctl(fd, DRM_IOCTL_GEM_CLOSE, &c) < 0)
		return -errno;
	return 0;
}
