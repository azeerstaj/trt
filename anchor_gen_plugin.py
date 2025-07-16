from trt_utils import *
from modules.torch_anchor import anchor_forward

anchor_plugin_name = "AnchorGenPlugin"
n_outputs = 1
numpy_dtype = np.float32

class AnchorGenPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    def __init__(self):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.num_outputs = n_outputs
        self.plugin_namespace = ""
        self.plugin_name = anchor_plugin_name
        self.plugin_version = "1"
        self.cuDevice = None
        self.map_sizes = [[200, 200], [100, 100], [50, 50], [25, 25], [13, 13]]
        self.sizes = ((32,), (64,), (128,), (256,), (512,))
        self.aspect_ratios = ((0.5, 1.0, 2.0),) * len(self.sizes)


    def get_capability_interface(self, type):
        # print("Capability")
        return self

    # Return Data Type
    def get_output_data_types(self, input_types):
        # print("Output dtypes")
        return [trt.DataType.FLOAT]#, trt.DataType.FLOAT]

    # inputs : shape of inputs
    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        output_dims = [trt.DimsExprs(2)]
        
        # img_width = exprBuilder.constant(inputs[0][-1])
        total_output_anchors = 0
        # print("\n\n\n\n")
        for map_size in self.map_sizes:
            print(f"Map size: {total_output_anchors}")
            total_output_anchors += map_size[0] * map_size[1] * 3 #len(self.sizes)
        # print("\n\n\n\n")
            
        output_dims[0][0] = exprBuilder.constant(total_output_anchors)
        output_dims[0][1] = exprBuilder.constant(4)
        print("Done calculating output dims")
        return output_dims

    # plugin input params, custom backend?
    def get_fields_to_serialize(self):
        return trt.PluginFieldCollection([])

    # depending on the plugin field, configure plugin
    def configure_plugin(self, inp, out):
        # print("Configure plugin")
        err, self.cuDevice = cuda.cuDeviceGet(0)

    # called when executing
    def on_shape_change(self, inp, out):
        # print("On shape change")
        err, self.cuDevice = cuda.cuDeviceGet(0)


    # Return true if plugin supports the format and datatype for the input/output indexed by pos.
    def supports_format_combination(self, pos, in_out, num_inputs):
        # print("pos", pos, "format", in_out[pos].desc.format, "type", in_out[pos].desc.type)
        return num_inputs == 6

    # The executed function when the plugin is called
    # workspace : the allowed gpu mem for this plugin
    # stream : cuda stream this plugin will run on
    # input_desc & output_desc : dims, format and type 
    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        img_dtype = trt.nptype(input_desc[0].type) # imgs

        # img
        imgs_mem = cp.cuda.UnownedMemory(
            inputs[0], volume(input_desc[0].dims) * cp.dtype(img_dtype).itemsize, self
        )

        # fmap1
        map1_mem = cp.cuda.UnownedMemory(
            inputs[1], volume(input_desc[1].dims) * cp.dtype(img_dtype).itemsize, self
        )

        # fmap2
        map2_mem = cp.cuda.UnownedMemory(
            inputs[2], volume(input_desc[2].dims) * cp.dtype(img_dtype).itemsize, self
        )

        # fmap3
        map3_mem = cp.cuda.UnownedMemory(
            inputs[3], volume(input_desc[3].dims) * cp.dtype(img_dtype).itemsize, self
        )

        # fmap4
        map4_mem = cp.cuda.UnownedMemory(
            inputs[4], volume(input_desc[4].dims) * cp.dtype(img_dtype).itemsize, self
        )

        # fmap5
        map5_mem = cp.cuda.UnownedMemory(
            inputs[5], volume(input_desc[5].dims) * cp.dtype(img_dtype).itemsize, self
        )

        anchors_mem = cp.cuda.UnownedMemory(
            outputs[0],
            volume(output_desc[0].dims) * cp.dtype(img_dtype).itemsize,
            self,
        )
        print("[anchor_gen] Device Mem Allocated.")

        imgs_ptr = cp.cuda.MemoryPointer(imgs_mem, 0)
        map1_ptr = cp.cuda.MemoryPointer(map1_mem, 0)
        map2_ptr = cp.cuda.MemoryPointer(map2_mem, 0)
        map3_ptr = cp.cuda.MemoryPointer(map3_mem, 0)
        map4_ptr = cp.cuda.MemoryPointer(map4_mem, 0)
        map5_ptr = cp.cuda.MemoryPointer(map5_mem, 0)
        anchors_ptr = cp.cuda.MemoryPointer(anchors_mem, 0)
        print("[anchor_gen] Pointers Initialized.")

        imgs_d = cp.ndarray(tuple(input_desc[0].dims), dtype=img_dtype, memptr=imgs_ptr)
        map1_d = cp.ndarray(tuple(input_desc[1].dims), dtype=img_dtype, memptr=map1_ptr)
        map2_d = cp.ndarray(tuple(input_desc[2].dims), dtype=img_dtype, memptr=map2_ptr)
        map3_d = cp.ndarray(tuple(input_desc[3].dims), dtype=img_dtype, memptr=map3_ptr)
        map4_d = cp.ndarray(tuple(input_desc[4].dims), dtype=img_dtype, memptr=map4_ptr)
        map5_d = cp.ndarray(tuple(input_desc[5].dims), dtype=img_dtype, memptr=map5_ptr)
        anchors_d = cp.ndarray(tuple(output_desc[0].dims), dtype=img_dtype, memptr=anchors_ptr)
        print("[anchor_gen] Arrays populated.")

        imgs_t = torch.as_tensor(imgs_d, device="cuda")
        map1_t = torch.as_tensor(map1_d, device="cuda")
        map2_t = torch.as_tensor(map2_d, device="cuda")
        map3_t = torch.as_tensor(map3_d, device="cuda")
        map4_t = torch.as_tensor(map4_d, device="cuda")
        map5_t = torch.as_tensor(map5_d, device="cuda")
        print("[anchor_gen] Torch populated.")

        out = anchor_forward(
            imgs_t, 
            [map1_t, map2_t, map3_t, map4_t, map5_t],
            self.sizes, self.aspect_ratios
        )#.view(-1)
        cp.copyto(anchors_d, cp.asarray(out))
        # cp.copyto(anchors_d, cp.reshape(cp.asarray(out), (-1,)))

        return 0


    def attach_to_context(self, context):
        return self.clone()

    def set_tactic(self, tactic):
        pass

    def clone(self):
        cloned_plugin = AnchorGenPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin


class AnchorGenPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = anchor_plugin_name
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection([])

    def create_plugin(self, name, fc, phase):
        return AnchorGenPlugin()

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
    my_plugin_creator = AnchorGenPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    # Create plugin object
    builder, network = create_network()
    plg_creator = plg_registry.get_creator(anchor_plugin_name, "1", "")
    
    plugin_fields_list = []

    pfc = trt.PluginFieldCollection(plugin_fields_list)
    plugin = plg_creator.create_plugin(anchor_plugin_name, pfc, trt.TensorRTPhase.BUILD)

    # Populate network
    inputX = network.add_input(name="image", dtype=trt.DataType.FLOAT, shape=trt.Dims(image_shape))
    inputMap1 = network.add_input(name="map1", dtype=trt.DataType.FLOAT, shape=trt.Dims(f1_shape))
    inputMap2 = network.add_input(name="map2", dtype=trt.DataType.FLOAT, shape=trt.Dims(f2_shape))
    inputMap3 = network.add_input(name="map3", dtype=trt.DataType.FLOAT, shape=trt.Dims(f3_shape))
    inputMap4 = network.add_input(name="map4", dtype=trt.DataType.FLOAT, shape=trt.Dims(f4_shape))
    inputMap5 = network.add_input(name="map5", dtype=trt.DataType.FLOAT, shape=trt.Dims(f5_shape))

    out = network.add_plugin_v3(
        [inputX, inputMap1, inputMap2, inputMap3, inputMap4, inputMap5],
        [],
        plugin
    )
    out.get_output(0).name = "anchors"
    network.mark_output(tensor=out.get_output(0))
    build_engine = engine_from_network((builder, network), CreateConfig(fp16= True if precision == np.float16 else False))

    map1 = np.random.random(f1_shape).astype(numpy_dtype)
    map2 = np.random.random(f2_shape).astype(numpy_dtype)
    map3 = np.random.random(f3_shape).astype(numpy_dtype)
    map4 = np.random.random(f4_shape).astype(numpy_dtype)
    map5 = np.random.random(f5_shape).astype(numpy_dtype)

    image = np.random.random(image_shape).astype(numpy_dtype)


    with TrtRunner(build_engine, "trt_runner")as runner:
        outputs = runner.infer(
            {
                "image": image, "map1": map1,
                "map2": map2, "map3": map3,
                "map4":map4, "map5":map5 
            }
        )['anchors']
        print("Outputs Shape:", outputs.shape)
        print("Outputs:", outputs[:10])

  