package main

import "fmt"

const mask uint64 = 9223372036854775807

type Op interface {
	Apply(x int64) int64
}

type AddOp struct{ k int64 }
type XorOp struct{ k int64 }
type MulOp struct{ k int64 }
type ShiftOp struct{ k int64 }

func (o AddOp) Apply(x int64) int64   { return int64((uint64(x) + uint64(o.k)) & mask) }
func (o XorOp) Apply(x int64) int64   { return int64((uint64(x) ^ uint64(o.k)) & mask) }
func (o MulOp) Apply(x int64) int64   { return int64((uint64(x) * uint64(o.k)) & mask) }
func (o ShiftOp) Apply(x int64) int64 { return int64((uint64(x) ^ (uint64(x) >> uint64(o.k))) & mask) }

func main() {
	ops := []Op{
		AddOp{k: 17},
		XorOp{k: 2654435761},
		MulOp{k: 16777619},
		ShiftOp{k: 13},
	}
	var acc int64 = 1234567
	for i := 0; i < 5000000; i++ {
		acc = ops[i&3].Apply(acc)
	}
	fmt.Println(acc)
}
