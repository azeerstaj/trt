from plugins.trt_utils import *
from plugins.anchor_gen import AnchorGenPluginCreator, plugin_name, numpy_dtype

if __name__ == "__main__":
    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)
    # Create context
    _, cudaCtx = cuda.cuCtxCreate(0, cuDevice)

    precision = np.float32

    image_shape = [1, 3, 800, 800]
    f1_shape = [1, 256, 50, 50]

    images = torch.randn(image_shape)
    f1 = torch.randn(f1_shape)

    # Register plugin creator
    plg_registry = trt.get_plugin_registry()
    my_plugin_creator = AnchorGenPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    # Create plugin object
    builder, network = create_network()
    plg_creator = plg_registry.get_creator(plugin_name, "1", "")
    
    plugin_fields_list = []

    pfc = trt.PluginFieldCollection(plugin_fields_list)
    plugin = plg_creator.create_plugin(plugin_name, pfc, trt.TensorRTPhase.BUILD)

    # Populate network
    inputX = network.add_input(name="image", dtype=trt.DataType.FLOAT, shape=trt.Dims(image_shape))
    inputY = network.add_input(name="fmaps", dtype=trt.DataType.FLOAT, shape=trt.Dims(f1_shape))

    out = network.add_plugin_v3([inputX, inputY], [], plugin)
    out.get_output(0).name = "anchors"
    network.mark_output(tensor=out.get_output(0))
    build_engine = engine_from_network((builder, network), CreateConfig(fp16= True if precision == np.float16 else False))

    fmaps = np.random.random(f1_shape).astype(numpy_dtype)
    image = np.random.random(image_shape).astype(numpy_dtype)


    with TrtRunner(build_engine, "trt_runner")as runner:
        outputs = runner.infer({"image": image, "fmaps":fmaps})['anchors']
        print("Outputs Shape:", outputs.shape)
        print("Outputs:", outputs[:10])

    checkCudaErrors(cuda.cuCtxDestroy(cudaCtx))