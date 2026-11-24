package main

import "fmt"

type FrameHdr struct {
	ctrl       uint16
	seq        uint32
	payloadLen uint16
}

func runCodec(iters int64) uint64 {
	var s uint64 = 987654321
	var acc uint64 = 0
	const mask uint64 = 9223372036854775807
	for i := int64(0); i < iters; i++ {
		s = (s + 1140071481932319848) & mask
		ver := uint16(s & 15)
		kind := uint16((s >> 4) & 15)
		flags := uint16((s >> 8) & 255)
		hdr := FrameHdr{
			ctrl:       ver | (kind << 4) | (flags << 8),
			seq:        uint32((s >> 16) & 4294967295),
			payloadLen: uint16((s >> 48) & 65535),
		}
		dVer := uint64(hdr.ctrl & 15)
		dKind := uint64((hdr.ctrl >> 4) & 15)
		dFlags := uint64((hdr.ctrl >> 8) & 255)
		mixed := (dVer << 24) ^ (dKind << 16) ^ (dFlags << 8) ^ uint64(hdr.seq) ^ uint64(hdr.payloadLen)
		acc = ((acc ^ mixed) * 16777619) & mask
	}
	return acc
}

func main() {
	fmt.Println(runCodec(20000000))
}
