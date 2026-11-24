package main

import (
	"encoding/json"
	"fmt"
)

type OutMsg struct {
	Type string `json:"type"`
	Seq  int64  `json:"seq"`
	Pick int64  `json:"pick"`
	Ok   bool   `json:"ok"`
}

func main() {
	sample := []byte(`{"type":"response.output_item.done","ok":true,"seq":42,"nums":[10,20,30,40,50],"item":{"id":"fc_123","name":"write_file","arguments":"{\"path\":\"src/main.tn\",\"content\":\"line1\\nline2 \\u4f60\\u597d\"}"}}`)
	var checksum int64 = 0
	for i := int64(0); i < 5000; i++ {
		var ev map[string]any
		if err := json.Unmarshal(sample, &ev); err != nil {
			panic(err)
		}
		t, _ := ev["type"].(string)
		seqF, _ := ev["seq"].(float64)
		seq := int64(seqF)
		var num2 int64 = 0
		if arr, ok := ev["nums"].([]any); ok && len(arr) > 2 {
			if f, ok := arr[2].(float64); ok {
				num2 = int64(f)
			}
		}
		out, err := json.Marshal(OutMsg{
			Type: t,
			Seq:  seq + i,
			Pick: num2,
			Ok:   true,
		})
		if err != nil {
			panic(err)
		}
		checksum += int64(len(out)) + num2
	}
	fmt.Println(checksum)
}
