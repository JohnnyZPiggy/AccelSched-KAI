package main

import "github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"

const shadowName = "accelsched-shadow"

type shadowPlugin struct{ captureSessions }

var _ framework.Plugin = (*shadowPlugin)(nil)
var _ framework.PluginBuilder = newShadowPlugin

func newShadowPlugin(_ framework.PluginArguments) framework.Plugin { return &shadowPlugin{} }
func (*shadowPlugin) Name() string                                 { return shadowName }
func (p *shadowPlugin) OnSessionOpen(s *framework.Session)         { p.openCapture(s) }
func (p *shadowPlugin) OnSessionClose(s *framework.Session)        { p.closeCapture(s) }
