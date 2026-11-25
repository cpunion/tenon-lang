package main

/*
#include <stdint.h>
static inline int64_t c_fast_hash(int64_t x) {
    return ((x * 6364136223846793005LL + 1442695040888963407LL) >> 17) ^ x;
}
int64_t go_cgo_hash(int64_t x) { return c_fast_hash(x); }
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

type Vec3 struct {
	X, Y, Z int64
}

func main() {
	// 1. 10,000 Goroutines work-stealing benchmark
	const N = 10000
	var total int64
	t0 := time.Now()
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		id := int64(i)
		go func(v int64) {
			defer wg.Done()
			var acc int64 = v
			for k := int64(0); k < 64; k++ {
				acc = (acc*1103515245 + 12345) & 0x7fffffff
			}
			atomic.AddInt64(&total, acc&255)
		}(id)
	}
	wg.Wait()
	dTasks := time.Since(t0)

	// 2. 1,000,000 C FFI calls via cgo inside a goroutine
	const FFI = 1000000
	t1 := time.Now()
	var ffiAcc int64
	var wg2 sync.WaitGroup
	wg2.Add(1)
	go func() {
		defer wg2.Done()
		var a int64 = 42
		for i := 0; i < FFI; i++ {
			a = int64(C.go_cgo_hash(C.int64_t(a + int64(i))))
		}
		ffiAcc = a
	}()
	wg2.Wait()
	dFfi := time.Since(t1)

	// 3. 50,000,000 mutable stack value struct loop
	const ITERS = 50000000
	t2 := time.Now()
	p := Vec3{X: 1, Y: 2, Z: 3}
	for i := int64(0); i < ITERS; i++ {
		p.X = p.X + p.Y + 1
		p.Y = p.Y ^ (p.X >> 3)
		p.Z = p.Z + p.Y - (i & 7)
	}
	dStruct := time.Since(t2)

	fmt.Printf("GO_BENCH gomaxprocs=%d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("GO_TASKS_10K ms=%.3f checksum=%d\n", float64(dTasks.Nanoseconds())/1e6, total)
	fmt.Printf("GO_CGO_1M ms=%.3f ns_per_call=%.2f checksum=%d\n", float64(dFfi.Nanoseconds())/1e6, float64(dFfi.Nanoseconds())/float64(FFI), ffiAcc)
	fmt.Printf("GO_MUT_STRUCT_50M ms=%.3f ns_per_iter=%.2f checksum=%d\n", float64(dStruct.Nanoseconds())/1e6, float64(dStruct.Nanoseconds())/float64(ITERS), p.X+p.Y+p.Z)
}
