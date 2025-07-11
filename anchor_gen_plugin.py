from trt_utils import *
from modules.torch_anchor import anchor_forward

# kernel_path = "modules/cuAnchor.cuh"

# with open(kernel_path, "r") as f:
#     template_kernel = f.read()

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
        self.sizes = [[32], [64], [128]]
        self.aspect_ratios = [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]


    def get_capability_interface(self, type):
        # print("Capability")
        return self

    # Return Data Type
    def get_output_data_types(self, input_types):
        # print("Output dtypes")
        return [trt.DataType.FLOAT]

    # inputs : shape of inputs
    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        output_dims = [trt.DimsExprs(2)]
        total_output_anchors = exprBuilder.operation(trt.DimensionOperation.PROD, inputs[1][-1], inputs[1][-2])
        total_output_anchors = exprBuilder.operation(trt.DimensionOperation.PROD, 
                                                    total_output_anchors, 
                                                    exprBuilder.constant(len(self.sizes)))

        output_dims[0][0] = total_output_anchors
        output_dims[0][1] = exprBuilder.constant(4)
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
        return num_inputs == 2

    # The executed function when the plugin is called
    # workspace : the allowed gpu mem for this plugin
    # stream : cuda stream this plugin will run on
    # input_desc & output_desc : dims, format and type 
    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        print("INFERENCE ...")
        img_dtype = trt.nptype(input_desc[0].type) # imgs

        imgs_mem = cp.cuda.UnownedMemory(
            inputs[0], volume(input_desc[0].dims) * cp.dtype(img_dtype).itemsize, self
        )

        fmaps_mem = cp.cuda.UnownedMemory(
            inputs[1], volume(input_desc[1].dims) * cp.dtype(img_dtype).itemsize, self
        )
        
        anchors_mem = cp.cuda.UnownedMemory(
            outputs[0],
            volume(output_desc[0].dims) * cp.dtype(img_dtype).itemsize,
            self,
        )
        print("Device Mem Allocated.")

        imgs_ptr = cp.cuda.MemoryPointer(imgs_mem, 0)
        fmaps_ptr = cp.cuda.MemoryPointer(fmaps_mem, 0)
        anchors_ptr = cp.cuda.MemoryPointer(anchors_mem, 0)
        print("Pointers Initialized.")

        imgs_d = cp.ndarray(tuple(input_desc[0].dims), dtype=img_dtype, memptr=imgs_ptr)
        fmaps_d = cp.ndarray(tuple(input_desc[1].dims), dtype=img_dtype, memptr=fmaps_ptr)
        anchors_d = cp.ndarray((volume(output_desc[0].dims)), dtype=img_dtype, memptr=anchors_ptr)
        print("Arrays populated.")

        imgs_t = torch.as_tensor(imgs_d, device="cuda")
        fmaps_t = torch.as_tensor(fmaps_d, device="cuda")
        print("Torch populated.")
        out = anchor_forward(imgs_t, fmaps_t, self.sizes, self.aspect_ratios).view(-1)
        cp.copyto(anchors_d, cp.asarray(out))

        return 0


    def attach_to_context(self, context):
        return self.clone()

    def set_tactic(self, tactic):
        pass

    def clone(self):
        cloned_plugin = AnchorGenPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def preprocess_input(self, image_height, image_width, feature_map_shapes, sizes, aspect_ratios):
        """
        NumPy version of the preprocessing function that returns NumPy arrays
        """

        base_anchors_list = []
        anchor_counts = []
        level_offsets = []
        
        total_base_anchors = 0
        for scale_set, ratio_set in zip(sizes, aspect_ratios):
            level_base_anchors = []
            for scale in scale_set:
                for ratio in ratio_set:
                    h = scale * np.sqrt(ratio)
                    w = scale / np.sqrt(ratio)
                    x1 = -w / 2
                    y1 = -h / 2
                    x2 = w / 2
                    y2 = h / 2
                    level_base_anchors.extend([x1, y1, x2, y2])
            
            num_anchors = len(level_base_anchors) // 4
            level_offsets.append(total_base_anchors)
            anchor_counts.append(num_anchors)
            total_base_anchors += num_anchors
            base_anchors_list.extend(level_base_anchors)

        # Convert to NumPy arrays
        base_anchors = np.array(base_anchors_list, dtype=np.float32).reshape(-1, 4)
        base_anchors = np.round(base_anchors)

        feature_map_info = []
        output_offsets = []
        total_output_anchors = 0

        for i, (feat_h, feat_w) in enumerate(feature_map_shapes):
            stride_h = image_height // feat_h
            stride_w = image_width // feat_w
            feature_map_info.extend([feat_h, feat_w, stride_h, stride_w])

            level_output_anchors = feat_h * feat_w * anchor_counts[i]
            total_output_anchors += level_output_anchors
            output_offsets.append(total_output_anchors)

        feature_map_info = np.array(feature_map_info, dtype=np.int32).reshape(-1, 4)
        anchor_counts = np.array(anchor_counts, dtype=np.int32)
        level_offsets = np.array(level_offsets, dtype=np.int32)
        output_offsets = np.array(output_offsets, dtype=np.int32)

        return base_anchors, feature_map_info, anchor_counts, level_offsets, output_offsets, total_output_anchors



class AnchorGenPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = anchor_plugin_name
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [
                # trt.PluginField("backend", np.array([]), trt.PluginFieldType.CHAR)
            ]
        )

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
    f1_shape = [1, 256, 50, 50]

    # f1_shape = [
    #     [1, 256, 50, 50],
    #     # [1, 256, 50, 50],
    #     # [1, 256, 50, 50],
    # ]

    # images = torch.randn(image_shape)
    # f1 = torch.randn(f1_shape)

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

  