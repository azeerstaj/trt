from trt_utils import *
from modules.torch_anchor import anchor_forward
from modules.rpn_head_forward import rpn_head_forward

# kernel_path = "modules/cuAnchor.cuh"

# with open(kernel_path, "r") as f:
#     template_kernel = f.read()

rpn_head_plugin_name = "RPNHeadPlugin"
n_outputs = 2
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
        return [trt.DataType.FLOAT, trt.DataType.FLOAT]

    # inputs : shape of inputs
    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        # 1st -> iC, iH, iW
        # 2nd -> fC, fH, fW
        output_dims = [trt.DimsExprs(3), trt.DimsExprs(3)]#, trt.DimsExprs(4)]

        output_dims[0][-1] = inputs[0][-1]
        # print("output_dims[0][-1]", output_dims[0][-1].get_constant_value())

        output_dims[0][-2] = inputs[0][-2]
        # print("output_dims[0][-2]", output_dims[0][-2].get_constant_value())

        output_dims[1][-1] = inputs[0][-1]
        # print("output_dims[1][-1]", output_dims[1][-1].get_constant_value())

        output_dims[1][-2] = inputs[0][-2]
        # print("output_dims[1][-2]", output_dims[1][-2].get_constant_value())

        # total_output_anchors = exprBuilder.operation(trt.DimensionOperation.PROD, inputs[0][-1], inputs[0][-2])
        total_output_anchors = exprBuilder.constant(3) # TODO : DYNAMIC NOT HARDCODED
        total_output_anchorsX4 = exprBuilder.operation(trt.DimensionOperation.PROD, total_output_anchors, 
                                                    exprBuilder.constant(4))

        output_dims[0][0] = total_output_anchors
        output_dims[1][0] = total_output_anchorsX4

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
        return num_inputs == 1

    # The executed function when the plugin is called
    # workspace : the allowed gpu mem for this plugin
    # stream : cuda stream this plugin will run on
    # input_desc & output_desc : dims, format and type 
    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        img_dtype = trt.nptype(input_desc[0].type) # imgs

        fmaps_mem = cp.cuda.UnownedMemory(
            inputs[0], volume(input_desc[0].dims) * cp.dtype(img_dtype).itemsize, self
        )

        cls_logits_mem = cp.cuda.UnownedMemory(
            outputs[0],
            volume(output_desc[0].dims) * cp.dtype(img_dtype).itemsize,
            self,
        )

        bbox_pred_mem = cp.cuda.UnownedMemory(
            outputs[1],
            volume(output_desc[1].dims) * cp.dtype(img_dtype).itemsize,
            self,
        )
        print("Device Mem Allocated.")

        fmaps_ptr = cp.cuda.MemoryPointer(fmaps_mem, 0)
        cls_logits_ptr = cp.cuda.MemoryPointer(cls_logits_mem, 0)
        bbox_pred_ptr = cp.cuda.MemoryPointer(bbox_pred_mem, 0)
        print("Pointers Initialized.")

        fmaps_d = cp.ndarray(tuple(input_desc[0].dims), dtype=img_dtype, memptr=fmaps_ptr)
        cls_logits_d = cp.ndarray((volume(output_desc[0].dims)), dtype=img_dtype, memptr=cls_logits_ptr)
        bbox_pred_d = cp.ndarray((volume(output_desc[1].dims)), dtype=img_dtype, memptr=bbox_pred_ptr)
        print("Arrays populated.")

        fmaps_t = torch.as_tensor(fmaps_d, device="cuda")
        # cls_logits_t = torch.as_tensor(cls_logits_d, device="cuda")
        # bbox_pred_t = torch.as_tensor(bbox_pred_d, device="cuda")
        print("Torch populated.")

        logits, bbox_reg = rpn_head_forward(x=[fmaps_t], weights=self.weights)#.cpu()
        # out = anchor_forward(imgs_t, fmaps_t, self.sizes, self.aspect_ratios).view(-1)
        print(logits[0].shape)
        print(bbox_reg[0].shape)
        print("dims:", output_desc[0].dims)
        # cp.copyto(cls_logits_d, cp.asarray(logits))
        cp.copyto(cls_logits_d, cp.reshape(cp.asarray(logits), (-1,)))
        cp.copyto(bbox_pred_d, cp.reshape(cp.asarray(bbox_reg), (-1,)))
        # cp.copyto(bbox_pred_d, cp.asarray(bbox_reg))

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
        self.field_names = trt.PluginFieldCollection(
            [
                # trt.PluginField("backend", np.array([]), trt.PluginFieldType.CHAR)
            ]
        )

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

    f1_shape = [1, 256, 50, 50]

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
    inputX = network.add_input(name="fmaps", dtype=trt.DataType.FLOAT, shape=trt.Dims(f1_shape))

    out = network.add_plugin_v3([inputX], [], plugin)

    out.get_output(0).name = "logits"
    out.get_output(1).name = "bbox_pred"

    network.mark_output(tensor=out.get_output(0))
    network.mark_output(tensor=out.get_output(1))

    build_engine = engine_from_network((builder, network), CreateConfig(fp16= True if precision == np.float16 else False))

    fmaps = np.random.random(f1_shape).astype(numpy_dtype)

    with TrtRunner(build_engine, "trt_runner")as runner:
        out = runner.infer({"fmaps":fmaps})
        print(out['logits'].shape)
        print(out['bbox_pred'].shape)
        # print("Outputs Shape:", outputs.shape)
        # print("Outputs:", outputs[:10])

  