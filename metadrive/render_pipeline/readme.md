### It is adapted from https://github.com/tobspr/RenderPipeline by @tobspr
### The Water effect is from: https://github.com/kergalym/RenderPipeline @kergalym

### How to integrate render-pipeline by modifying engine core
1. copy rpcore/rplibs/rplugin/config/data/effect to project
2. rename ```from rpcore``` to ```from metadrive.rpcore```
3. rename ```from rplibs``` to ```from metadrive.rplibs```
4. rename ```from rpplugins``` to ```from metadrive.rpplugins```
5. In engine core:

```
        self._render_pipeline = RenderPipeline()
        self._render_pipeline.pre_showbase_init()
        
        showbase.__init__()
        self._render_pipeline.create(self)
```

6. disable multi-thread rendering
7. disable pbrpipe
8. disable engine_core ```self.render.setShaderAuto()```
9. change ```assert __package__ == "rpcore.native"``` to ```assert __package__ == "metadrive.rpcore.native"```
10. change ```plugin_class = "rpplugins.{}.plugin".format(plugin_id)```
    to ```plugin_class = "metadrive.rpplugins.{}.plugin".format(plugin_id)```
11. disable display logo

```
# Display logo
            # if self.mode == RENDER_MODE_ONSCREEN and (not self.global_config["debug"]):
            #     if self.global_config["show_logo"]:
            #         self._window_logo = attach_logo(self)
            #     self._loading_logo = attach_cover_image(
            #         window_width=self.get_size()[0], window_height=self.get_size()[1]
            #     )
            #     for i in range(5):
            #         self.graphicsEngine.renderFrame()
            #     self.taskMgr.add(self.remove_logo, "remove _loading_logo in first frame")
```

12. add ```self._render_pipeline.daytime_mgr.time = "12:08"```
13. disabale skybox and all lights
14. disable compress-texture!
15. main_camera: ```self.camera = engine.camera```
16. disable gltf loader
17. disable all s_frgb


### Performance Bossting related configure (The lower the better, List not complete)
1. culling_max_distance
2. culling_slice_width

AO:
3. AO blur_quality
4. SSAO Sample Sequence
5. SSVO Sample Sequence
6. ALCHEMY Sample Sequence
7. UE4AO Sample Sequence

cloud:
8. Raymarch Steps

color correction:
9. Chromatic Aberration Samples (a lot)

fxaa:
10. FXAA Quality

vxgi:
- Voxel Grid Resolution
volumetrics:
- Sample count
