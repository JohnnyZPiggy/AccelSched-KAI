package main

import "github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"

const shadowName = "accelsched-shadow"

type shadowPlugin struct{}

var _ framework.Plugin = (*shadowPlugin)(nil)
var _ framework.PluginBuilder = newShadowPlugin

func newShadowPlugin(_ framework.PluginArguments) framework.Plugin { return &shadowPlugin{} }
func (*shadowPlugin) Name() string                                 { return shadowName }
func (*shadowPlugin) OnSessionOpen(*framework.Session)             {}
func (*shadowPlugin) OnSessionClose(*framework.Session)            {}
