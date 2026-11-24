package main

import "fmt"

func computeMix(iters int64) uint64 {
	var x uint64 = 123456789
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
	fmt.Println(computeMix(100000000))
}
