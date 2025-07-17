from trt_utils import *
from anchor_gen_plugin import AnchorGenPluginCreator, anchor_plugin_name, numpy_dtype
from rpn_head_plugin import RPNHeadPluginCreator, rpn_head_plugin_name
from modules.rpn_forward import rpn_forward
from collections import namedtuple

torch.manual_seed(0)

rpn_plugin_name = "RPNPlugin"
n_outputs = 2
numpy_dtype = np.float32

ImageList = namedtuple("ImageList", ["tensors", "image_sizes"])
class RPNPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    def __init__(self):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.num_outputs = n_outputs
        self.plugin_namespace = ""
        self.plugin_name = rpn_plugin_name
        self.plugin_version = "1"
        self.cuDevice = None
        self.max_proposals = 1000

    def get_capability_interface(self, type):
        # print("Capability")
        return self

    # Return Data Type
    def get_output_data_types(self, input_types):
        # print("Output dtypes")
        return [trt.DataType.FLOAT, trt.DataType.INT32]

    # inputs : shape of inputs
    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        output_dims = [trt.DimsExprs(2), trt.DimsExprs(1)]
        max_props = exprBuilder.constant(self.max_proposals)

        output_dims[0][0] = max_props
        output_dims[0][1] = exprBuilder.constant(4)
        output_dims[1][0] = exprBuilder.constant(1)
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
        return num_inputs == 12 # imgs, fmaps, anchors, 

    # The executed function when the plugin is called
    # workspace : the allowed gpu mem for this plugin
    # stream : cuda stream this plugin will run on
    # input_desc & output_desc : dims, format and type 
    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        img_dtype = trt.nptype(input_desc[0].type) # imgs
        active_rows_dtype = trt.nptype(output_desc[1].type) # imgs
        print("Img dtype:", img_dtype)
        print("Active Rows dtype:", active_rows_dtype)
        print("Img Dims:", input_desc[0].dims)
        print("Active Rows dims:", output_desc[1].dims)
        print("Input Len", len(inputs), "Output Len:", len(outputs))

        imgs_mem = cp.cuda.UnownedMemory(
            inputs[0], volume(input_desc[0].dims) * cp.dtype(img_dtype).itemsize, self
        )

        anchor_mem = cp.cuda.UnownedMemory(
            inputs[1], volume(input_desc[1].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_1_mem = cp.cuda.UnownedMemory(
            inputs[2], volume(input_desc[2].dims) * cp.dtype(img_dtype).itemsize, self
        )
        bbox_reg_1_mem = cp.cuda.UnownedMemory(
            inputs[3], volume(input_desc[3].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_2_mem = cp.cuda.UnownedMemory(
            inputs[4], volume(input_desc[4].dims) * cp.dtype(img_dtype).itemsize, self
        )
        bbox_reg_2_mem = cp.cuda.UnownedMemory(
            inputs[5], volume(input_desc[5].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_3_mem = cp.cuda.UnownedMemory(
            inputs[6], volume(input_desc[6].dims) * cp.dtype(img_dtype).itemsize, self
        )
        bbox_reg_3_mem = cp.cuda.UnownedMemory(
            inputs[7], volume(input_desc[7].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_4_mem = cp.cuda.UnownedMemory(
            inputs[8], volume(input_desc[8].dims) * cp.dtype(img_dtype).itemsize, self
        )
        bbox_reg_4_mem = cp.cuda.UnownedMemory(
            inputs[9], volume(input_desc[9].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_5_mem = cp.cuda.UnownedMemory(
            inputs[10], volume(input_desc[10].dims) * cp.dtype(img_dtype).itemsize, self
        )
        bbox_reg_5_mem = cp.cuda.UnownedMemory(
            inputs[11], volume(input_desc[11].dims) * cp.dtype(img_dtype).itemsize, self
        )


        boxes_mem = cp.cuda.UnownedMemory(
            outputs[0], volume(output_desc[0].dims) * cp.dtype(img_dtype).itemsize, self,
        )

        active_rows_mem = cp.cuda.UnownedMemory(
            outputs[1], volume(output_desc[1].dims) * cp.dtype(active_rows_dtype).itemsize, self,
        )
        print("Device Mem Allocated.")

        imgs_ptr = cp.cuda.MemoryPointer(imgs_mem, 0)
        anchor_ptr = cp.cuda.MemoryPointer(anchor_mem, 0)

        cls_logits_1_ptr = cp.cuda.MemoryPointer(cls_logits_1_mem, 0)
        bbox_reg_1_ptr = cp.cuda.MemoryPointer(bbox_reg_1_mem, 0)
        cls_logits_2_ptr = cp.cuda.MemoryPointer(cls_logits_2_mem, 0)
        bbox_reg_2_ptr = cp.cuda.MemoryPointer(bbox_reg_2_mem, 0)
        cls_logits_3_ptr = cp.cuda.MemoryPointer(cls_logits_3_mem, 0)
        bbox_reg_3_ptr = cp.cuda.MemoryPointer(bbox_reg_3_mem, 0)
        cls_logits_4_ptr = cp.cuda.MemoryPointer(cls_logits_4_mem, 0)
        bbox_reg_4_ptr = cp.cuda.MemoryPointer(bbox_reg_4_mem, 0)
        cls_logits_5_ptr = cp.cuda.MemoryPointer(cls_logits_5_mem, 0)
        bbox_reg_5_ptr = cp.cuda.MemoryPointer(bbox_reg_5_mem, 0)

        boxes_ptr = cp.cuda.MemoryPointer(boxes_mem, 0)
        active_rows_ptr = cp.cuda.MemoryPointer(active_rows_mem, 0)
        print("Pointers Initialized.")

        imgs_d = cp.ndarray(tuple(input_desc[0].dims), dtype=img_dtype, memptr=imgs_ptr)
        anchor_d = cp.ndarray(tuple(input_desc[1].dims), dtype=img_dtype, memptr=anchor_ptr)
        cls_logits_1_d = cp.ndarray(tuple(input_desc[2].dims), dtype=img_dtype, memptr=cls_logits_1_ptr)
        bbox_reg_1_d = cp.ndarray(tuple(input_desc[3].dims), dtype=img_dtype, memptr=bbox_reg_1_ptr)
        cls_logits_2_d = cp.ndarray(tuple(input_desc[4].dims), dtype=img_dtype, memptr=cls_logits_2_ptr)
        bbox_reg_2_d = cp.ndarray(tuple(input_desc[5].dims), dtype=img_dtype, memptr=bbox_reg_2_ptr)
        cls_logits_3_d = cp.ndarray(tuple(input_desc[6].dims), dtype=img_dtype, memptr=cls_logits_3_ptr)
        bbox_reg_3_d = cp.ndarray(tuple(input_desc[7].dims), dtype=img_dtype, memptr=bbox_reg_3_ptr)
        cls_logits_4_d = cp.ndarray(tuple(input_desc[8].dims), dtype=img_dtype, memptr=cls_logits_4_ptr)
        bbox_reg_4_d = cp.ndarray(tuple(input_desc[9].dims), dtype=img_dtype, memptr=bbox_reg_4_ptr)
        cls_logits_5_d = cp.ndarray(tuple(input_desc[10].dims), dtype=img_dtype, memptr=cls_logits_5_ptr)
        bbox_reg_5_d = cp.ndarray(tuple(input_desc[11].dims), dtype=img_dtype, memptr=bbox_reg_5_ptr)
       
        boxes_d = cp.ndarray((volume(output_desc[0].dims)), dtype=img_dtype, memptr=boxes_ptr)
        active_rows_d = cp.ndarray((volume(output_desc[1].dims)), dtype=active_rows_dtype, memptr=active_rows_ptr)
        print("Arrays populated.")

        # Simulated Objectness & Proposals
        imgs_t = torch.as_tensor(imgs_d, device="cuda")
        anchor_t = torch.as_tensor(anchor_d, device="cuda")
        cls_logits_1_t = torch.as_tensor(cls_logits_1_d, device="cuda")
        bbox_reg_1_t = torch.as_tensor(bbox_reg_1_d, device="cuda")
        cls_logits_2_t = torch.as_tensor(cls_logits_2_d, device="cuda")
        bbox_reg_2_t = torch.as_tensor(bbox_reg_2_d, device="cuda")
        cls_logits_3_t = torch.as_tensor(cls_logits_3_d, device="cuda")
        bbox_reg_3_t = torch.as_tensor(bbox_reg_3_d, device="cuda")
        cls_logits_4_t = torch.as_tensor(cls_logits_4_d, device="cuda")
        bbox_reg_4_t = torch.as_tensor(bbox_reg_4_d, device="cuda")
        cls_logits_5_t = torch.as_tensor(cls_logits_5_d, device="cuda")
        bbox_reg_5_t = torch.as_tensor(bbox_reg_5_d, device="cuda")
        print("Torch populated.")

        print("[enqueue] imgs shape:", imgs_t.shape, "values:", imgs_t.view(-1)[:5])
        # print("[enqueue] fmaps shape:", fmaps_t.shape, "values:", fmaps_t.view(-1)[:5])
        # print("[enqueue] anchors shape:", anchors_t.shape, "values:", anchors_t.view(-1)[:5])
        # print("[enqueue] objectness shape:", objectness_t.shape, "values:", objectness_t.view(-1)[:5])
        # print("[enqueue] proposals shape:", proposals_t.shape, "values:", proposals_t.view(-1)[:5])

        img_list = ImageList(imgs_t, [imgs_t.shape[-2:]])
        out = rpn_forward(
            img_list,
            [anchor_t], 
            [cls_logits_1_t, 
            cls_logits_2_t,
            cls_logits_3_t,
            cls_logits_4_t,
            cls_logits_5_t],
            [bbox_reg_1_t,
            bbox_reg_2_t,
            bbox_reg_3_t,
            bbox_reg_4_t,
            bbox_reg_5_t],
        )

        print("len(output):", len(out))
        print("Output Shape:", out[0].shape)  # e.g., (N, 4)
        max_proposals = self.max_proposals

        boxes_np = out[0]
        num_proposals = boxes_np.shape[0]
        print("num_proposals:", num_proposals)  # e.g., (N, 4)

        if num_proposals < max_proposals:
            pad = np.zeros((max_proposals - num_proposals, boxes_np.shape[1]), dtype=boxes_np.dtype)
            boxes_np = np.concatenate([boxes_np, pad], axis=0)
        elif num_proposals > max_proposals:
            boxes_np = boxes_np[:max_proposals]
        print("boxes_np.shape:", boxes_np.shape)

        # Now boxes_np has shape (max_proposals, 4)
        cp.copyto(boxes_d, cp.reshape(cp.asarray(boxes_np), (-1,)))
        cp.copyto(active_rows_d, cp.asarray([num_proposals]))
        
        return 0


    def attach_to_context(self, context):
        return self.clone()

    def set_tactic(self, tactic):
        pass

    def clone(self):
        cloned_plugin = RPNPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin


class RPNPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = rpn_plugin_name
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection([])

    def create_plugin(self, name, fc, phase):
        return RPNPlugin()


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

    plg_registry.register_creator(rpn_plugin_creator, "")
    plg_registry.register_creator(rpn_head_plugin_creator, "")
    plg_registry.register_creator(anchor_plugin_creator, "")

    # Create plugin object
    builder, network = create_network()
    rpn_plg_creator = plg_registry.get_creator(rpn_plugin_name, "1", "")
    rpn_head_plg_creator = plg_registry.get_creator(rpn_head_plugin_name, "1", "")
    anchor_plg_creator = plg_registry.get_creator(anchor_plugin_name, "1", "")
    
    plugin_fields_list = []
    pfc = trt.PluginFieldCollection(plugin_fields_list)

    # Plugins
    rpn_plugin = rpn_plg_creator.create_plugin(rpn_plugin_name, pfc, trt.TensorRTPhase.BUILD)
    rpn_head_plugin = rpn_head_plg_creator.create_plugin(rpn_head_plugin_name, pfc, trt.TensorRTPhase.BUILD)
    anchor_plugin = anchor_plg_creator.create_plugin(anchor_plugin_name, pfc, trt.TensorRTPhase.BUILD)

    # Populate network
    inputImage = network.add_input(name="image", dtype=trt.DataType.FLOAT, shape=trt.Dims(image_shape))
    inputFmap1 = network.add_input(name="f1", dtype=trt.DataType.FLOAT, shape=trt.Dims(f1_shape))
    inputFmap2 = network.add_input(name="f2", dtype=trt.DataType.FLOAT, shape=trt.Dims(f2_shape))
    inputFmap3 = network.add_input(name="f3", dtype=trt.DataType.FLOAT, shape=trt.Dims(f3_shape))
    inputFmap4 = network.add_input(name="f4", dtype=trt.DataType.FLOAT, shape=trt.Dims(f4_shape))
    inputFmap5 = network.add_input(name="f5", dtype=trt.DataType.FLOAT, shape=trt.Dims(f5_shape))

    # layers outputs
    anchor_out = network.add_plugin_v3(
        [
            inputImage, inputFmap1,
            inputFmap2, inputFmap3,
            inputFmap4, inputFmap5 
        ],
        [],
        anchor_plugin
    )

    rpn_head_out = network.add_plugin_v3(
        [
            inputFmap1, inputFmap2, 
            inputFmap3, inputFmap4,
            inputFmap5
        ],
        [], 
        rpn_head_plugin
    )

    out = network.add_plugin_v3(
        [
            inputImage,
            anchor_out.get_output(0), 
            # cls_logits                # bbox_pred
            rpn_head_out.get_output(0), rpn_head_out.get_output(1), # map1 outputs
            rpn_head_out.get_output(2), rpn_head_out.get_output(3), # map2 outputs
            rpn_head_out.get_output(4), rpn_head_out.get_output(5), # map3 outputs
            rpn_head_out.get_output(6), rpn_head_out.get_output(7), # map4 outputs
            rpn_head_out.get_output(8), rpn_head_out.get_output(9), # map5 outputs
        ], 
        [], rpn_plugin
    )

    out.get_output(0).name = "boxes"
    out.get_output(1).name = "active_rows"
    network.mark_output(tensor=out.get_output(0))
    network.mark_output(tensor=out.get_output(1))

    load = True
    if load:
        build_engine = engine_from_path("engines/rpn_1.engine")
        print("Engine loaded.")
    else:
        build_engine = engine_from_network((builder, network), CreateConfig(fp16=True if precision == np.float16 else False))
        save_engine(build_engine, "engines/rpn_1.engine")
        print("Engine built and saved.")

    image = torch.randn(image_shape).numpy().astype(numpy_dtype)
    map1 = torch.randn(f1_shape).numpy().astype(numpy_dtype)
    map2 = torch.randn(f2_shape).numpy().astype(numpy_dtype)
    map3 = torch.randn(f3_shape).numpy().astype(numpy_dtype)
    map4 = torch.randn(f4_shape).numpy().astype(numpy_dtype)
    map5 = torch.randn(f5_shape).numpy().astype(numpy_dtype)
    
    with TrtRunner(build_engine, "trt_runner")as runner:
        outputs = runner.infer(
            {
                "image": image, "f1":map1, 
                "f2":map2, "f3":map3,
                "f4":map4, "f5":map5
            }
        )
        for k in outputs.keys():
            print(f"Outputs[{k}].shape:", outputs[k].shape)
            print(f"Outputs[{k}].dtype:", outputs[k].dtype)
            print(f"type(Outputs[{k}]):", type(outputs[k]))
            # print(f"Outputs[{k}]:", outputs[k][0][0][:10])

    checkCudaErrors(cuda.cuCtxDestroy(cudaCtx))