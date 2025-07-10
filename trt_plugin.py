from trt_utils import *
# from anchor_gen import AnchorGenPluginCreator, plugin_name, numpy_dtype
from modules.rpn_forward import rpn_forward
from collections import namedtuple

kernel_path = "modules/cuAnchor.cuh"

with open(kernel_path, "r") as f:
    template_kernel = f.read()

plugin_name = "RPNPlugin"
n_outputs = 1
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
        self.plugin_name = plugin_name
        self.plugin_version = "1"
        self.cuDevice = None

    def get_capability_interface(self, type):
        # print("Capability")
        return self

    # Return Data Type
    def get_output_data_types(self, input_types):
        # print("Output dtypes")
        return [trt.DataType.FLOAT]

    # inputs : shape of inputs
    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        output_dims = [trt.DimsExprs(1)]
        # max_rpn_size = exprBuilder.declare_size_tensor(1,
        #                                                exprBuilder.constant(500),
        #                                                exprBuilder.constant(1000))

        output_dims[0][0] = exprBuilder.constant(1000)
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
        return num_inputs == 5 # imgs, fmaps, anchors, 

    # The executed function when the plugin is called
    # workspace : the allowed gpu mem for this plugin
    # stream : cuda stream this plugin will run on
    # input_desc & output_desc : dims, format and type 
    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        print("INFERENCE ...")
        # print("LEN OF  INPUTS:", len(input_desc))
        # print("LEN OF OUTPUTS:", len(output_desc))

        img_dtype = trt.nptype(input_desc[0].type) # imgs

        imgs_mem = cp.cuda.UnownedMemory(
            inputs[0], volume(input_desc[0].dims) * cp.dtype(img_dtype).itemsize, self
        )

        fmaps_mem = cp.cuda.UnownedMemory(
            inputs[1], volume(input_desc[1].dims) * cp.dtype(img_dtype).itemsize, self
        )
        
        anchors_mem = cp.cuda.UnownedMemory(
            inputs[2], volume(input_desc[2].dims) * cp.dtype(img_dtype).itemsize, self,
        )

        objectness_mem = cp.cuda.UnownedMemory(
            inputs[3], volume(input_desc[3].dims) * cp.dtype(img_dtype).itemsize, self,
        )

        proposals_mem = cp.cuda.UnownedMemory(
            inputs[4], volume(input_desc[4].dims) * cp.dtype(img_dtype).itemsize, self,
        )

        boxes_mem = cp.cuda.UnownedMemory(
            outputs[0], volume(output_desc[0].dims) * cp.dtype(img_dtype).itemsize, self,
        )
        print("Device Mem Allocated.")

        imgs_ptr = cp.cuda.MemoryPointer(imgs_mem, 0)
        fmaps_ptr = cp.cuda.MemoryPointer(fmaps_mem, 0)
        anchors_ptr = cp.cuda.MemoryPointer(anchors_mem, 0)
        objectness_ptr = cp.cuda.MemoryPointer(objectness_mem, 0)
        proposals_ptr = cp.cuda.MemoryPointer(proposals_mem, 0)
        boxes_ptr = cp.cuda.MemoryPointer(boxes_mem, 0)
        print("Pointers Initialized.")

        imgs_d = cp.ndarray(tuple(input_desc[0].dims), dtype=img_dtype, memptr=imgs_ptr)
        fmaps_d = cp.ndarray(tuple(input_desc[1].dims), dtype=img_dtype, memptr=fmaps_ptr)
        anchors_d = cp.ndarray(tuple(input_desc[2].dims), dtype=img_dtype, memptr=anchors_ptr)
        objectness_d = cp.ndarray(tuple(input_desc[3].dims), dtype=img_dtype, memptr=objectness_ptr)
        proposals_d = cp.ndarray(tuple(input_desc[4].dims), dtype=img_dtype, memptr=proposals_ptr)
        boxes_d = cp.ndarray((volume(output_desc[0].dims)), dtype=img_dtype, memptr=boxes_ptr)
        print("Arrays populated.")

        imgs_t = torch.as_tensor(imgs_d, device="cuda")#.squeeze(0)
        fmaps_t = torch.as_tensor(fmaps_d, device="cuda")#.squeeze(0)
        anchors_t = torch.as_tensor(anchors_d, device="cuda").squeeze(0)
        objectness_t = torch.as_tensor(objectness_d, device="cuda").squeeze(0)
        proposals_t = torch.as_tensor(proposals_d, device="cuda").squeeze(0)
        # boxes_t = torch.as_tensor(boxes_d, device="cuda")
        print("Torch populated.")

        print("imgs shape:", imgs_t.shape)
        print("fmaps shape:", fmaps_t.shape)
        print("anchors shape:", anchors_t.shape)
        print("objectness shape:", objectness_t.shape)
        print("proposals shape:", proposals_t.shape)

        img_list = ImageList(imgs_t, [imgs_t.shape[:-2]])
        out = rpn_forward(img_list, fmaps_t,
                          [anchors_t], 
                          [objectness_t],# .cpu().numpy().tolist()[0],
                          [proposals_t]) # .cpu().numpy().tolist())

        print("len(output) :", len(out))
        print("Output Shape:", out[0].shape)
        # print(out)
        # cp.copyto(boxes_d, cp.asarray(out))

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
        self.name = plugin_name
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [
                # trt.PluginField("backend", np.array([]), trt.PluginFieldType.CHAR)
            ]
        )

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
    f1_shape = [1, 256, 50, 50]
    batch_size = 1

    features = [torch.randn(f1_shape)]

    # Register plugin creator
    plg_registry = trt.get_plugin_registry()
    my_plugin_creator = RPNPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    # Create plugin object
    builder, network = create_network()
    plg_creator = plg_registry.get_creator(plugin_name, "1", "")
    
    plugin_fields_list = []

    pfc = trt.PluginFieldCollection(plugin_fields_list)
    plugin = plg_creator.create_plugin(plugin_name, pfc, trt.TensorRTPhase.BUILD)

    objectness = []
    pred_bbox_delta = []

    for feat in features: # features
        N, C, H, W = feat.shape
        objectness.append(torch.randn(N, 3, H, W))
        pred_bbox_delta.append(torch.randn(N, 3 * 4, H, W))

    # Simulate anchors for each image
    anchors = []
    for _ in range(batch_size):
        all_anchors = []
        for feat in features:
            _, _, H, W = feat.shape
            A = 3 # num_anchors
            total = H * W * A
            anchors_per_feat = torch.rand(total, 4) * 800  # random box coords
            all_anchors.append(anchors_per_feat)
        anchors.append(torch.cat(all_anchors, dim=0))

    anchors = np.array(anchors).astype(numpy_dtype)
    objectness = np.array(objectness).astype(numpy_dtype)
    pred_bbox_delta = np.array(pred_bbox_delta).astype(numpy_dtype)

    # Populate network
    inputImage = network.add_input(name="image", dtype=trt.DataType.FLOAT, shape=trt.Dims(image_shape))
    inputFmaps = network.add_input(name="fmaps", dtype=trt.DataType.FLOAT, shape=trt.Dims(f1_shape))
    inputAnchors = network.add_input(name="anchors", dtype=trt.DataType.FLOAT, shape=trt.Dims(anchors.shape))
    inputObjectness = network.add_input(name="objectness", dtype=trt.DataType.FLOAT, shape=trt.Dims(objectness.shape))
    inputBbox = network.add_input(name="pred_bbox_delta", dtype=trt.DataType.FLOAT, shape=trt.Dims(pred_bbox_delta.shape))

    out = network.add_plugin_v3([inputImage, inputFmaps,
                                inputAnchors, inputObjectness,
                                inputBbox], [], plugin)

    out.get_output(0).name = "boxes"
    network.mark_output(tensor=out.get_output(0))
    build_engine = engine_from_network((builder, network), CreateConfig(fp16= True if precision == np.float16 else False))

    image = torch.randn(image_shape).numpy().astype(numpy_dtype)
    fmaps = torch.randn(f1_shape).numpy().astype(numpy_dtype)

    with TrtRunner(build_engine, "trt_runner")as runner:
        outputs = runner.infer({"image": image, 
                                "fmaps":fmaps,
                                "anchors":anchors,
                                "objectness":objectness,
                                "pred_bbox_delta":pred_bbox_delta})
        # print(outputs)#['boxes'])

    checkCudaErrors(cuda.cuCtxDestroy(cudaCtx))