from trt_utils import *
from anchor_gen_plugin import AnchorGenPluginCreator, anchor_plugin_name
from rpn_head_plugin import RPNHeadPluginCreator, rpn_head_plugin_name
from rpn_plugin import RPNPluginCreator, rpn_plugin_name
from roi_plugin import MScaleRoIPluginCreator, roi_plugin_name

torch.manual_seed(0)
numpy_dtype = np.float32

if __name__ == "__main__":
    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)
    # Create context
    _, cudaCtx = cuda.cuCtxCreate(0, cuDevice)

    precision = np.float32
    image_shape = [1, 3, 800, 800]
    f1_shape = [1, 256, 200, 200]
    f2_shape = [1, 256, 100, 100]
    f3_shape = [1, 256, 50, 50]
    f4_shape = [1, 256, 25, 25]
    f5_shape = [1, 256, 13, 13]

    # Register plugin creator
    plg_registry = trt.get_plugin_registry()
    rpn_plugin_creator = RPNPluginCreator()
    rpn_head_plugin_creator = RPNHeadPluginCreator()
    anchor_plugin_creator = AnchorGenPluginCreator()
    roi_plugin_creator = MScaleRoIPluginCreator()

    plg_registry.register_creator(rpn_plugin_creator, "")
    plg_registry.register_creator(rpn_head_plugin_creator, "")
    plg_registry.register_creator(anchor_plugin_creator, "")
    plg_registry.register_creator(roi_plugin_creator, "")

    rpn_plg_creator = plg_registry.get_creator(rpn_plugin_name, "1", "")
    rpn_head_plg_creator = plg_registry.get_creator(rpn_head_plugin_name, "1", "")
    anchor_plg_creator = plg_registry.get_creator(anchor_plugin_name, "1", "")
    roi_plg_creator = plg_registry.get_creator(roi_plugin_name, "1", "")
    
    plugin_fields_list = []
    pfc = trt.PluginFieldCollection(plugin_fields_list)

    # Plugins
    rpn_plugin = rpn_plg_creator.create_plugin(rpn_plugin_name, pfc, trt.TensorRTPhase.BUILD)
    rpn_head_plugin = rpn_head_plg_creator.create_plugin(rpn_head_plugin_name, pfc, trt.TensorRTPhase.BUILD)
    anchor_plugin = anchor_plg_creator.create_plugin(anchor_plugin_name, pfc, trt.TensorRTPhase.BUILD)
    roi_plugin = roi_plg_creator.create_plugin(roi_plugin_name, pfc, trt.TensorRTPhase.BUILD)

    build_engine = engine_from_path("engines/frcnn_noback_1.engine")
    print("Engine loaded.")
    
    image = torch.randn(image_shape).numpy().astype(numpy_dtype)
    map1 = torch.randn(f1_shape).numpy().astype(numpy_dtype)
    map2 = torch.randn(f2_shape).numpy().astype(numpy_dtype)
    map3 = torch.randn(f3_shape).numpy().astype(numpy_dtype)
    map4 = torch.randn(f4_shape).numpy().astype(numpy_dtype)
    map5 = torch.randn(f5_shape).numpy().astype(numpy_dtype)

    with TrtRunner(build_engine, "trt_runner") as runner:
        outputs = runner.infer(
            {
                "image": image, "f1":map1, 
                "f2":map2, "f3":map3,
                "f4":map4, "f5":map5
            }
        )
        for k in outputs.keys():
            print(f"Outputs[{k}].shape:", outputs[k].shape)
            # print(f"Outputs[{k}]:", outputs[k][0][0][:10])
        print(f"Outputs[active_rows_out]:", outputs["active_rows_out"])

    checkCudaErrors(cuda.cuCtxDestroy(cudaCtx))