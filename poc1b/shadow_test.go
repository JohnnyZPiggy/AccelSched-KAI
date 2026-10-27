package main

import (
	"reflect"
	"runtime"
	"testing"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/plugins"
)

// Registry tests deliberately run serially in this disposable test process.
// No production default name is registered by these tests. No test invokes
// main, RunApp, Run, OpenSession, scheduling, or any Kubernetes client.
func TestG1APICompatibility(t *testing.T) {
	if runtime.Version() != "go1.26.8" {
		t.Fatalf("exact go1.26.8 required, got %s", runtime.Version())
	}
	args := framework.PluginArguments{"probe": "unchanged"}
	p := newShadowPlugin(args)
	if p == nil || p.Name() != shadowName {
		t.Fatal("constructor does not produce the named shadow plugin")
	}
	// Empty hooks must neither dereference a Session nor register callbacks.
	p.OnSessionOpen(nil)
	p.OnSessionClose(nil)
	if !reflect.DeepEqual(args, framework.PluginArguments{"probe": "unchanged"}) {
		t.Fatal("constructor mutated arguments")
	}
}

func TestG2RegistryPreservation(t *testing.T) {
	previous, existed := framework.GetPluginBuilder(shadowName)
	if existed {
		t.Cleanup(func() { framework.RegisterPluginBuilder(shadowName, previous) })
	}
	framework.RegisterPluginBuilder(shadowName, newShadowPlugin)
	plugins.InitDefaultPlugins()
	builder, found := framework.GetPluginBuilder(shadowName)
	if !found || builder == nil {
		t.Fatal("default initialization lost the external builder")
	}
	p := builder(framework.PluginArguments{})
	if p == nil || p.Name() != shadowName {
		t.Fatal("retained builder does not construct the registered plugin")
	}
	t.Log("external builder retained after official InitDefaultPlugins; not enabled in a Session")
}

type collisionPlugin struct {
	shadowPlugin
	name string
}

func (p *collisionPlugin) Name() string { return p.name }

func TestG3RegistryOverwrite(t *testing.T) {
	const name = "accelsched-shadow-test-only-collision"
	previous, existed := framework.GetPluginBuilder(name)
	if existed {
		t.Cleanup(func() { framework.RegisterPluginBuilder(name, previous) })
	}
	first := func(framework.PluginArguments) framework.Plugin { return &collisionPlugin{name: "first"} }
	second := func(framework.PluginArguments) framework.Plugin { return &collisionPlugin{name: "second"} }
	framework.RegisterPluginBuilder(name, first)
	builder, found := framework.GetPluginBuilder(name)
	if !found || builder == nil || builder(nil).Name() != "first" {
		t.Fatal("first builder not registered")
	}
	framework.RegisterPluginBuilder(name, second)
	builder, found = framework.GetPluginBuilder(name)
	if !found || builder == nil || builder(nil).Name() != "second" {
		t.Fatal("same-name registration did not replace the earlier constructor")
	}
	t.Log("last registration wins for the unique disposable test name; no built-in name overwritten")
}
