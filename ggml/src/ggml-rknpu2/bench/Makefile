# RK3588 rknn_matmul micro-benchmarks. Requires librknnrt + rknn headers under
# /usr/include/rknn. The NPU is single-tenant: stop other NPU users before running.
CFLAGS = -O2 -I/usr/include/rknn
LIBS   = -lrknnrt

all: mm_peak mm_concurrent

mm_peak: mm_peak.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

mm_concurrent: mm_concurrent.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS) -lpthread

clean:
	rm -f mm_peak mm_concurrent
