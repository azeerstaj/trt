from trt_utils import *
from modules.torch_anchor import anchor_forward
from modules.rpn_head_forward import rpn_head_forward

rpn_head_plugin_name = "RPNHeadPlugin"
n_outputs = 10
numpy_dtype = np.float32

class RPNHeadPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    def __init__(self):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)


        self.num_outputs = n_outputs
        self.plugin_namespace = ""
        self.plugin_name = rpn_head_plugin_name
        self.plugin_version = "1"
        self.cuDevice = None
        self.sizes = [[32], [64], [128]]
        self.weights = torch.load("weights/fasterrcnn1.pt", weights_only=True, map_location='cuda')


    def get_capability_interface(self, type):
        # print("Capability")
        return self

    # Return Data Type
    def get_output_data_types(self, input_types):
        # print("Output dtypes")
        return [
                trt.DataType.FLOAT, trt.DataType.FLOAT,
                trt.DataType.FLOAT, trt.DataType.FLOAT,
                trt.DataType.FLOAT, trt.DataType.FLOAT,
                trt.DataType.FLOAT, trt.DataType.FLOAT,
                trt.DataType.FLOAT, trt.DataType.FLOAT
            ]

    # inputs : shape of inputs
    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        output_dims = [
            # 3, fH, fW       # 12, fH, fW
            trt.DimsExprs(3), trt.DimsExprs(3), # 1st map
            trt.DimsExprs(3), trt.DimsExprs(3), # 2nd map
            trt.DimsExprs(3), trt.DimsExprs(3), # 3rd map
            trt.DimsExprs(3), trt.DimsExprs(3), # 4th map
            trt.DimsExprs(3), trt.DimsExprs(3), # 5th map
        ]
        # print("Output Dims:", len(output_dims))

        total_output_anchors = exprBuilder.constant(3) # TODO : DYNAMIC NOT HARDCODED
        total_output_anchorsX4 = exprBuilder.operation(
            trt.DimensionOperation.PROD, 
            total_output_anchors, 
            exprBuilder.constant(4)
        )
        # print("Total Output Anchors:", total_output_anchors, total_output_anchorsX4)

        for i in range(len(output_dims) // 2):
            output_dims[i * 2][-1] = output_dims[i * 2 + 1][-1] = inputs[i][-1]
            output_dims[i * 2][-2] = output_dims[i * 2 + 1][-2] = inputs[i][-2]

            output_dims[i * 2][0] = total_output_anchors
            output_dims[i * 2 + 1][0] = total_output_anchorsX4

        print("Returning output dims:", len(output_dims))
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
        # print("Support Combination")
        return num_inputs == 5

    # The executed function when the plugin is called
    # workspace : the allowed gpu mem for this plugin
    # stream : cuda stream this plugin will run on
    # input_desc & output_desc : dims, format and type 
    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        # Inputs
        img_dtype = trt.nptype(input_desc[0].type) # imgs

        map1_mem = cp.cuda.UnownedMemory(
            inputs[0], volume(input_desc[0].dims) * cp.dtype(img_dtype).itemsize, self
        )

        map2_mem = cp.cuda.UnownedMemory(
            inputs[1], volume(input_desc[1].dims) * cp.dtype(img_dtype).itemsize, self
        )

        map3_mem = cp.cuda.UnownedMemory(
            inputs[2], volume(input_desc[2].dims) * cp.dtype(img_dtype).itemsize, self
        )

        map4_mem = cp.cuda.UnownedMemory(
            inputs[3], volume(input_desc[3].dims) * cp.dtype(img_dtype).itemsize, self
        )

        map5_mem = cp.cuda.UnownedMemory(
            inputs[4], volume(input_desc[4].dims) * cp.dtype(img_dtype).itemsize, self
        )

        # Outputs
        cls_logits_1_mem = cp.cuda.UnownedMemory(
            outputs[0], volume(output_desc[0].dims) * cp.dtype(img_dtype).itemsize, self
        )

        bbox_pred_1_mem = cp.cuda.UnownedMemory(
            outputs[1], volume(output_desc[1].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_2_mem = cp.cuda.UnownedMemory(
            outputs[2], volume(output_desc[2].dims) * cp.dtype(img_dtype).itemsize, self
        )

        bbox_pred_2_mem = cp.cuda.UnownedMemory(
            outputs[3], volume(output_desc[3].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_3_mem = cp.cuda.UnownedMemory(
            outputs[4], volume(output_desc[4].dims) * cp.dtype(img_dtype).itemsize, self
        )

        bbox_pred_3_mem = cp.cuda.UnownedMemory(
            outputs[5], volume(output_desc[5].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_4_mem = cp.cuda.UnownedMemory(
            outputs[6], volume(output_desc[6].dims) * cp.dtype(img_dtype).itemsize, self
        )

        bbox_pred_4_mem = cp.cuda.UnownedMemory(
            outputs[7], volume(output_desc[7].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_5_mem = cp.cuda.UnownedMemory(
            outputs[8], volume(output_desc[8].dims) * cp.dtype(img_dtype).itemsize, self
        )

        bbox_pred_5_mem = cp.cuda.UnownedMemory(
            outputs[9], volume(output_desc[9].dims) * cp.dtype(img_dtype).itemsize, self
        )
        print("Device Mem Allocated.")

        map1_ptr = cp.cuda.MemoryPointer(map1_mem, 0)
        map2_ptr = cp.cuda.MemoryPointer(map2_mem, 0)
        map3_ptr = cp.cuda.MemoryPointer(map3_mem, 0)
        map4_ptr = cp.cuda.MemoryPointer(map4_mem, 0)
        map5_ptr = cp.cuda.MemoryPointer(map5_mem, 0)

        cls_logits_1_ptr = cp.cuda.MemoryPointer(cls_logits_1_mem, 0)
        bbox_pred_1_ptr = cp.cuda.MemoryPointer(bbox_pred_1_mem, 0)

        cls_logits_2_ptr = cp.cuda.MemoryPointer(cls_logits_2_mem, 0)
        bbox_pred_2_ptr = cp.cuda.MemoryPointer(bbox_pred_2_mem, 0)

        cls_logits_3_ptr = cp.cuda.MemoryPointer(cls_logits_3_mem, 0)
        bbox_pred_3_ptr = cp.cuda.MemoryPointer(bbox_pred_3_mem, 0)

        cls_logits_4_ptr = cp.cuda.MemoryPointer(cls_logits_4_mem, 0)
        bbox_pred_4_ptr = cp.cuda.MemoryPointer(bbox_pred_4_mem, 0)

        cls_logits_5_ptr = cp.cuda.MemoryPointer(cls_logits_5_mem, 0)
        bbox_pred_5_ptr = cp.cuda.MemoryPointer(bbox_pred_5_mem, 0)
        print("Pointers Initialized.")

        map1_d = cp.ndarray(tuple(input_desc[0].dims), dtype=img_dtype, memptr=map1_ptr)
        map2_d = cp.ndarray(tuple(input_desc[1].dims), dtype=img_dtype, memptr=map2_ptr)
        map3_d = cp.ndarray(tuple(input_desc[2].dims), dtype=img_dtype, memptr=map3_ptr)
        map4_d = cp.ndarray(tuple(input_desc[3].dims), dtype=img_dtype, memptr=map4_ptr)
        map5_d = cp.ndarray(tuple(input_desc[4].dims), dtype=img_dtype, memptr=map5_ptr)

        cls_logits_1_d = cp.ndarray((volume(output_desc[0].dims)), dtype=img_dtype, memptr=cls_logits_1_ptr)
        bbox_pred_1_d = cp.ndarray((volume(output_desc[1].dims)), dtype=img_dtype, memptr=bbox_pred_1_ptr)

        cls_logits_2_d = cp.ndarray((volume(output_desc[2].dims)), dtype=img_dtype, memptr=cls_logits_2_ptr)
        bbox_pred_2_d = cp.ndarray((volume(output_desc[3].dims)), dtype=img_dtype, memptr=bbox_pred_2_ptr)

        cls_logits_3_d = cp.ndarray((volume(output_desc[4].dims)), dtype=img_dtype, memptr=cls_logits_3_ptr)
        bbox_pred_3_d = cp.ndarray((volume(output_desc[5].dims)), dtype=img_dtype, memptr=bbox_pred_3_ptr)

        cls_logits_4_d = cp.ndarray((volume(output_desc[6].dims)), dtype=img_dtype, memptr=cls_logits_4_ptr)
        bbox_pred_4_d = cp.ndarray((volume(output_desc[7].dims)), dtype=img_dtype, memptr=bbox_pred_4_ptr)

        cls_logits_5_d = cp.ndarray((volume(output_desc[8].dims)), dtype=img_dtype, memptr=cls_logits_5_ptr)
        bbox_pred_5_d = cp.ndarray((volume(output_desc[9].dims)), dtype=img_dtype, memptr=bbox_pred_5_ptr)
        print("Arrays populated.")

        map1_t = torch.as_tensor(map1_d, device="cuda")
        map2_t = torch.as_tensor(map2_d, device="cuda")
        map3_t = torch.as_tensor(map3_d, device="cuda")
        map4_t = torch.as_tensor(map4_d, device="cuda")
        map5_t = torch.as_tensor(map5_d, device="cuda")
        print("Torch populated.")

        logits, bbox_reg = rpn_head_forward(
            x=[map1_t, map2_t, map3_t, map4_t, map5_t],
            weights=self.weights
        )

        print("len(logits):", len(logits))#.shape)
        print("len(bbox_reg):", len(bbox_reg))#.shape)
        # print("len(logits[0]):", len(logits[0]))#.shape)
        # print("len(bbox_reg[0]):", len(bbox_reg[0]))#.shape)
        # print("len(logits[0][0]):", len(logits[0][0]))#.shape)
        # print("len(bbox_reg[0][0]):", len(bbox_reg[0][0]))#.shape)

        # print("logits[0][0].shape:", logits[0][0].shape)
        # print("bbox_reg[0][0].shape:", bbox_reg[0][0].shape)

        # print("Plugin Output[0][0] Dims:", output_desc[0].dims)
        # print("Plugin Output[0][1] Dims:", output_desc[1].dims)
        # print("Plugin Output[1][0] Dims:", output_desc[2].dims)
        # print("Plugin Output[1][1] Dims:", output_desc[3].dims)

        print("1st:")
        print(f" - logits[0].shape:{logits[0].shape}, bbox_reg[0].shape:{bbox_reg[0].shape}")
        # print(f" - value:{logits[0][0][0][:5]}, value:{bbox_reg[0][0][0][:5]}")
        cp.copyto(cls_logits_1_d, cp.reshape(cp.asarray(logits[0]), (-1,)))
        cp.copyto(bbox_pred_1_d, cp.reshape(cp.asarray(bbox_reg[0]), (-1,)))

        print("2nd:")
        print(f" - logits[1].shape:{logits[1].shape}, bbox_reg[1].shape:{bbox_reg[1].shape}")
        # print(f" - value:{logits[1][0][0][:5]}, value:{bbox_reg[1][0][0][:5]}")
        cp.copyto(cls_logits_2_d, cp.reshape(cp.asarray(logits[1]), (-1,)))
        cp.copyto(bbox_pred_2_d, cp.reshape(cp.asarray(bbox_reg[1]), (-1,)))

        print("3rd:")
        print(f" - logits[2].shape:{logits[2].shape}, bbox_reg[2].shape:{bbox_reg[2].shape}")
        # print(f" - value:{logits[2][0][0][:5]}, value:{bbox_reg[2][0][0][:5]}")
        cp.copyto(cls_logits_3_d, cp.reshape(cp.asarray(logits[2]), (-1,)))
        cp.copyto(bbox_pred_3_d, cp.reshape(cp.asarray(bbox_reg[2]), (-1,)))

        print("4th:")
        print(f" - logits[3].shape:{logits[3].shape}, bbox_reg[3].shape:{bbox_reg[3].shape}")
        # print(f" - value:{logits[3][0][0][:5]}, value:{bbox_reg[3][0][0][:5]}")
        cp.copyto(cls_logits_4_d, cp.reshape(cp.asarray(logits[3]), (-1,)))
        cp.copyto(bbox_pred_4_d, cp.reshape(cp.asarray(bbox_reg[3]), (-1,)))

        print("5th:")
        print(f" - logits[4].shape:{logits[4].shape}, bbox_reg[4].shape:{bbox_reg[4].shape}")
        # print(f" - value:{logits[4][0][0][:5]}, value:{bbox_reg[4][0][0][:5]}")
        cp.copyto(cls_logits_5_d, cp.reshape(cp.asarray(logits[4]), (-1,)))
        cp.copyto(bbox_pred_5_d, cp.reshape(cp.asarray(bbox_reg[4]), (-1,)))

        return 0


    def attach_to_context(self, context):
        return self.clone()

    def set_tactic(self, tactic):
        pass

    def clone(self):
        cloned_plugin = RPNHeadPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin


class RPNHeadPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = rpn_head_plugin_name
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection([])

    def create_plugin(self, name, fc, phase):
        return RPNHeadPlugin()

if __name__ == "__main__":
    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)
    # Create context
    _, cudaCtx = cuda.cuCtxCreate(0, cuDevice)

    precision = np.float32
    f1_shape = [1, 256, 200, 200]
    f2_shape = [1, 256, 100, 100]
    f3_shape = [1, 256, 50, 50]
    f4_shape = [1, 256, 25, 25]
    f5_shape = [1, 256, 13, 13]

    # Register plugin creator
    plg_registry = trt.get_plugin_registry()
    my_plugin_creator = RPNHeadPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    # Create plugin object
    builder, network = create_network()
    plg_creator = plg_registry.get_creator(rpn_head_plugin_name, "1", "")
    
    plugin_fields_list = []

    pfc = trt.PluginFieldCollection(plugin_fields_list)
    plugin = plg_creator.create_plugin(rpn_head_plugin_name, pfc, trt.TensorRTPhase.BUILD)

    # Populate network
    inputMap1 = network.add_input(name="map1", dtype=trt.DataType.FLOAT, shape=trt.Dims(f1_shape))
    inputMap2 = network.add_input(name="map2", dtype=trt.DataType.FLOAT, shape=trt.Dims(f2_shape))
    inputMap3 = network.add_input(name="map3", dtype=trt.DataType.FLOAT, shape=trt.Dims(f3_shape))
    inputMap4 = network.add_input(name="map4", dtype=trt.DataType.FLOAT, shape=trt.Dims(f4_shape))
    inputMap5 = network.add_input(name="map5", dtype=trt.DataType.FLOAT, shape=trt.Dims(f5_shape))

    out = network.add_plugin_v3(
        [
            inputMap1, inputMap2,
            inputMap3, inputMap4, 
            inputMap5
        ],
        [], plugin
    )

    print("out.num_outputs:", out.num_outputs)

    for i in range(n_outputs // 2):
        out.get_output(i * 2).name = f"logits_{i}"
        out.get_output(i * 2 + 1).name = f"bbox_pred_{i}"

        network.mark_output(tensor=out.get_output(i * 2))
        network.mark_output(tensor=out.get_output(i * 2 + 1))

    build_engine = engine_from_network(
        (builder, network), 
        CreateConfig(fp16= True if precision == np.float16 else False)
    )

    map1 = np.random.random(f1_shape).astype(numpy_dtype)
    map2 = np.random.random(f2_shape).astype(numpy_dtype)
    map3 = np.random.random(f3_shape).astype(numpy_dtype)
    map4 = np.random.random(f4_shape).astype(numpy_dtype)
    map5 = np.random.random(f5_shape).astype(numpy_dtype)

    with TrtRunner(build_engine, "trt_runner")as runner:
        out = runner.infer(
            {
                "map1": map1,
                "map2": map2,
                "map3": map3,
                "map4": map4,
                "map5": map5
            }
        )

        print("Output  Keys:", out.keys())
        for k in out.keys():
            print(f"Output {k} Shape:", out[k].shape)
            print(f"Output {k} Values:", out[k][0][0][:10], "\n\n")