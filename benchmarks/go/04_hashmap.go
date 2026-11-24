package main

import (
	"fmt"
	"strconv"
)

func main() {
	const n = 50000
	m := make(map[string]string)
	for i := 0; i < n; i++ {
		k := "key_" + strconv.Itoa(i)
		v := "val_" + strconv.Itoa(i*3)
		m[k] = v
	}
	for u := 0; u < 10000; u++ {
		m["key_"+strconv.Itoa(u)] = "upd_" + strconv.Itoa(u)
	}
	var checksum int64 = 0
	for q := 0; q < n; q++ {
		if v, ok := m["key_"+strconv.Itoa(q)]; ok {
			checksum += int64(len(v))
		}
	}
	for d := 0; d < n; d += 2 {
		delete(m, "key_"+strconv.Itoa(d))
	}
	for q := 0; q < n; q++ {
		if v, ok := m["key_"+strconv.Itoa(q)]; ok {
			checksum += int64(len(v))
		}
	}
	checksum += int64(len(m))
	fmt.Println(checksum)
}
