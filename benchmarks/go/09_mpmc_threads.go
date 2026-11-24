package main

import "fmt"

func workerMix(x uint64, iters int64) uint64 {
	var acc uint64 = 0
	const mask uint64 = 9223372036854775807
	for i := int64(0); i < iters; i++ {
		x = (x + 1140071481932319848) & mask
		z := x
		z = ((z ^ (z >> 30)) * 490278349) & mask
		z = ((z ^ (z >> 27)) * 16777619) & mask
		z = z ^ (z >> 31)
		acc = (acc ^ z) & mask
	}
	return acc
}

func main() {
	seeds := [4]uint64{100000007, 200000033, 300000007, 400000009}
	ch := make(chan uint64, 4)
	for _, s := range seeds {
		go func(seed uint64) {
			ch <- workerMix(seed, 25000000)
		}(s)
	}
	const mask uint64 = 9223372036854775807
	var total uint64 = 0
	for i := 0; i < 4; i++ {
		total = (total ^ <-ch) & mask
	}
	fmt.Println(total)
}
