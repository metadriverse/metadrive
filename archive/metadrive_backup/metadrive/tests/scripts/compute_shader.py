from panda3d.core import Shader, Texture, ConfigVariableBool, NodePath, ShaderAttrib, PNMImage
from direct.showbase.ShowBase import ShowBase, loadPrcFileData

loadPrcFileData("", "textures-power-2 none")
immutableTextureStore = ConfigVariableBool("gl-immutable-texture-storage", False)
immutableTextureStore.setValue(True)
print("immutable texture storage", immutableTextureStore.getValue())

base = ShowBase()

# Setup the textures
myTex1 = Texture()
myTex2 = Texture()
myTex2.setup_2d_texture(200, 100, Texture.T_unsigned_byte, Texture.F_rgba8)
myTex1.read("test.png")
# myTex1.setFormat(Texture.TT2dTexture)
# myTex1.setComponentType(Texture.T_unsigned_byte)

# Create a dummy node and apply the shader to it
shader = Shader.load_compute(Shader.SL_GLSL, "compute_shader.glsl")
dummy = NodePath("dummy")
dummy.set_shader(shader)
dummy.set_shader_input("fromTex", myTex1)
dummy.set_shader_input("toTex", myTex2)

# Retrieve the underlying ShaderAttrib
sattr = dummy.get_attrib(ShaderAttrib)

# Dispatch the compute shader, right now!
base.graphicsEngine.dispatch_compute((32, 32, 1), sattr, base.win.get_gsg())

# Store the output
base.graphicsEngine.extractTextureData(myTex2, base.win.get_gsg())
frame = PNMImage()
myTex2.store(frame)
frame.write("test_compute_shader.png")
base.destroy()
