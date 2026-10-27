// PoC-1B B1/B2 compile/registry probe. Build this scheduler; NEVER execute it
// during acceptance. RunApp starts real client/server/scheduler setup.
package main

import (
	"fmt"
	"os"

	"github.com/kai-scheduler/KAI-scheduler/cmd/scheduler/app"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
)

var _ func() error = app.RunApp // External API compatibility, not invocation.

func main() {
	framework.RegisterPluginBuilder(shadowName, newShadowPlugin)
	if err := app.RunApp(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
