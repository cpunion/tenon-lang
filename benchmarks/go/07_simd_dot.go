package main

import "fmt"

func main() {
	const N = 1024
	const REPS = 50000
	var a [N][4]float64
	var b [N][4]float64

	for i := 0; i < N; i++ {
		a[i] = [4]float64{
			float64((i+1)&15) * 0.25,
			float64((i+3)&15) * 0.25,
			float64((i+5)&15) * 0.25,
			float64((i+7)&15) * 0.25,
		}
		b[i] = [4]float64{
			float64((i*3+2)&15) * 0.25,
			float64((i*3+4)&15) * 0.25,
			float64((i*3+6)&15) * 0.25,
			float64((i*3+8)&15) * 0.25,
		}
	}

	total := 0.0
	for r := 0; r < REPS; r++ {
		for j := 0; j < N; j++ {
			x := a[j]
			y := b[(j+r)&1023]
			total += x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3]
		}
	}

	fmt.Println(int64(total * 16.0))
}
