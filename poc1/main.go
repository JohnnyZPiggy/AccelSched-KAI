// PoC-1A offline CPU-only Go observer; standard library only.
//
// Usage: go run . -trace trace.json -snapshot ../fixtures/01_gpu_family.json -out NEW_DIR
// Validate/reduce BEFORE creating output. Exit 0: audit bundle written (possibly
// TRACE_INCOMPLETE); exit 1: schema/I/O failure, no success claim; exit 2: usage.
// Trace completeness is in trace-result.json, not inferred from CLI exit success.
//
// Contract/source references (read-only, never imported as Go dependencies):
// KAI 374cbb796ecff07c4a9140e59286f3e3a280e84b:
// pkg/scheduler/framework/{event,statement,session}.go,
// pkg/scheduler/actions/allocate/allocate.go,
// pkg/binder/controllers/bindrequest_controller.go.
// Event has Task only. Statement rollback truncates operations; Discard ignores
// undo errors; Commit can fail after partial effects. Session.BindPod/Cache.Bind
// is separate from BindRequest and Binder completion. No actual integration.
//
// Run: go test ./...; go test -race ./... (requires supported CGO/C compiler);
// go vet ./... . POC1_EVIDENCE_DIR=NEW_PARENT retains tiny synthetic bundles from
// the six test groups outside the source tree. No Python import/runtime is used.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
)

func readInput(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, (1<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(b) > 1<<20 {
		return nil, fmt.Errorf("input exceeds 1 MiB")
	}
	return b, nil
}

func run(args []string, stderr io.Writer) int {
	f := flag.NewFlagSet("poc1", flag.ContinueOnError)
	f.SetOutput(stderr)
	trace := f.String("trace", "", "strict trace v0.1 local JSON file")
	snapshot := f.String("snapshot", "", "unchanged strict PoC-0 snapshot local JSON file")
	out := f.String("out", "", "new output directory, never overwritten")
	if err := f.Parse(args); err != nil {
		return 2
	}
	if f.NArg() != 0 || *trace == "" || *snapshot == "" || *out == "" {
		f.Usage()
		return 2
	}
	tr, err := readInput(*trace)
	if err == nil {
		var sn []byte
		sn, err = readInput(*snapshot)
		if err == nil {
			var files map[string][]byte
			files, err = bundle(tr, sn)
			if err == nil {
				err = writeBundle(*out, files)
			}
		}
	}
	if err != nil {
		fmt.Fprintln(stderr, "INVALID_INPUT_OR_IO:", err)
		return 1
	}
	return 0
}

func main() { os.Exit(run(os.Args[1:], os.Stderr)) }
