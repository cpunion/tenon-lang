// examples/dogfood/go_cgo_demo/main.go — Real Go `cgo` integration calling a verified Tenon
// static library (`libquant.a` + `quant.h` + auto-generated `quant_cgo.go`).
package main

import (
	"fmt"
	"os"
)

func main() {
	riskOk := Quant_verify_order_risk(50000, 2, 10, 200000)
	riskBlocked := Quant_verify_order_risk(50000, 5, 10, 200000)
	ema := Quant_ema_bps(10000, 11000, 2000)
	sig := Quant_simd_signal_score(100.0, 102.0, 105.0, 110.0)
	pnl := Quant_backtest_pnl(100000, 32)

	fmt.Printf("Go<-Tenon C-ABI: riskOk=%d riskBlocked=%d ema=%d sig=%d pnl=%d\n",
		riskOk, riskBlocked, ema, sig, pnl)
	if riskOk != 1 || riskBlocked != 0 || ema != 10200 || sig != 10590 {
		os.Exit(1)
	}
}
