package main

import "fmt"

func sieveCount(limit int) int64 {
	flags := make([]int64, limit+1)
	for i := range flags {
		flags[i] = 1
	}
	flags[0] = 0
	flags[1] = 0
	for p := 2; p*p <= limit; p++ {
		if flags[p] == 1 {
			for m := p * p; m <= limit; m += p {
				flags[m] = 0
			}
		}
	}
	var count int64 = 0
	for i := 2; i <= limit; i++ {
		count += flags[i]
	}
	return count
}

func main() {
	var total int64 = 0
	for r := 0; r < 5; r++ {
		total += sieveCount(1000000)
	}
	fmt.Println(total)
}
