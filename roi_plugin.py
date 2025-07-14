from trt_utils import *
from modules.roi_forward import (
    multiscale_roi_align_forward,
    box_head_forward,
    box_predictor_forward
)

# Multi-Scale RoI Plugin.
mscale_roi_plugin_name = "MScaleRoIPlugin"
n_outputs = 2
numpy_dtype = np.float32

class MScaleRoIPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    def __init__(self):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.num_outputs = n_outputs
        self.plugin_namespace = ""
        self.plugin_name = mscale_roi_plugin_name
        self.plugin_version = "1"
        self.cuDevice = None
        self.output_size = 7
        self.out_cls_score = 91
        self.out_bbox_pred = 364
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
        # inputs #0 : (B, C, H, W)
        # inputs #2 : (n_boxes, 4)
        # inputs #1 : (H, W)

        # outputs #0 : n_boxes
        # outputs #1 : C
        # outputs #2 : self.output_size
        # outputs #3 : self.output_size
        output_dims = [trt.DimsExprs(2), trt.DimsExprs(2)]#, trt.DimsExprs(4)]

        output_dims[0][0] = exprBuilder.constant(1)
        output_dims[1][0] = exprBuilder.constant(1)
        output_dims[0][1] = exprBuilder.constant(self.out_cls_score) # cls
        output_dims[1][1] = exprBuilder.constant(self.out_bbox_pred)       # bbox

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
        return num_inputs == 3

    # The executed function when the plugin is called
    # workspace : the allowed gpu mem for this plugin
    # stream : cuda stream this plugin will run on
    # input_desc & output_desc : dims, format and type 
    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        img_dtype = trt.nptype(input_desc[0].type) # imgs
        roi_dtype = trt.nptype(output_desc[0].type) # imgs

        # fmaps
        fmaps_mem = cp.cuda.UnownedMemory(
            inputs[0],
            volume(input_desc[0].dims) * cp.dtype(img_dtype).itemsize,
            self
        )

        # images
        images_mem = cp.cuda.UnownedMemory(
            inputs[1],
            volume(input_desc[1].dims) * cp.dtype(img_dtype).itemsize,
            self
        )

        # boxes
        boxes_mem = cp.cuda.UnownedMemory(
            inputs[2],
            volume(input_desc[2].dims) * cp.dtype(img_dtype).itemsize,
            self
        )

        # cls reg
        cls_reg_mem = cp.cuda.UnownedMemory(
            outputs[0],
            volume(output_desc[0].dims) * cp.dtype(roi_dtype).itemsize,
            self,
        )

        # cls reg
        bbox_predictor_mem = cp.cuda.UnownedMemory(
            outputs[1],
            volume(output_desc[1].dims) * cp.dtype(roi_dtype).itemsize,
            self,
        )
        print("Device Mem Allocated.")

        fmaps_ptr = cp.cuda.MemoryPointer(fmaps_mem, 0)
        boxes_ptr = cp.cuda.MemoryPointer(boxes_mem, 0)
        images_ptr = cp.cuda.MemoryPointer(images_mem, 0)
        cls_reg_ptr = cp.cuda.MemoryPointer(cls_reg_mem, 0)
        bbox_predictor_ptr = cp.cuda.MemoryPointer(bbox_predictor_mem, 0)
        print("Pointers Initialized.")

        fmaps_d = cp.ndarray(tuple(input_desc[0].dims), dtype=img_dtype, memptr=fmaps_ptr)
        boxes_d = cp.ndarray(tuple(input_desc[1].dims), dtype=img_dtype, memptr=boxes_ptr)
        images_d = cp.ndarray(tuple(input_desc[2].dims), dtype=img_dtype, memptr=images_ptr)
        cls_reg_d = cp.ndarray((volume(output_desc[0].dims)), dtype=img_dtype, memptr=cls_reg_ptr)
        bbox_predictor_d = cp.ndarray((volume(output_desc[1].dims)), dtype=img_dtype, memptr=bbox_predictor_ptr)
        print("Arrays populated.")

        fmaps_t = torch.as_tensor(fmaps_d, device="cuda")
        boxes_t = torch.as_tensor(boxes_d, device="cuda")
        images_t = torch.as_tensor(images_d, device="cuda")
        print("Torch populated.")
        
        print("[enqueue] MAPS:", fmaps_t.shape, "values:", fmaps_t.view(-1)[:5])
        print("[enqueue] BOXES:", boxes_t.shape, "values:", boxes_t.view(-1)[:5])
        print("[enqueue] IMGS:", images_t.shape, "values:", images_t.view(-1)[:5])

        # Postprocessing
        fmap_num = 0
        fmaps_t_dict = dict()
        for m in fmaps_t:
            fmaps_t_dict[f"f_{fmap_num}"] = m.unsqueeze(0)
            fmap_num += 1

        boxes_t = [b.cuda().unsqueeze(0) for b in boxes_t]
        image_sizes = [(i.shape[-2], i.shape[-1]) for i in images_t]

        box_features = multiscale_roi_align_forward(boxes=boxes_t, 
                                           features=fmaps_t_dict, 
                                           image_sizes=image_sizes,
                                           output_size=self.output_size)#.cpu().numpy()

        # box_features
        box_features = box_head_forward(box_features, self.weights)
        class_logits, box_regression = box_predictor_forward(box_features, self.weights)

        print("Final box_features Size:", box_features.shape)
        print("Class logits Size:", class_logits.shape)
        print("Box regression Size:", box_regression.shape)
        # cp.copyto(cls_logits_d, cp.asarray(logits))

        cp.copyto(cls_reg_d, cp.reshape(cp.asarray(class_logits), (-1,)))
        cp.copyto(bbox_predictor_d, cp.reshape(cp.asarray(box_regression), (-1,)))
        return 0


    def attach_to_context(self, context):
        return self.clone()

    def set_tactic(self, tactic):
        pass

    def clone(self):
        cloned_plugin = MScaleRoIPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin


class RPNHeadPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = mscale_roi_plugin_name
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [
                # trt.PluginField("backend", np.array([]), trt.PluginFieldType.CHAR)
            ]
        )

    def create_plugin(self, name, fc, phase):
        return MScaleRoIPlugin()

if __name__ == "__main__":
    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)
    # Create context
    _, cudaCtx = cuda.cuCtxCreate(0, cuDevice)

    precision = np.float32

    f1_shape = [1, 256, 50, 50]
    image_shape = [1, 3, 800, 800]
    boxes = torch.tensor(
            [
                [50, 60, 200, 220], 
                [300, 300, 450, 450], 
                [100, 100, 300, 350],
                [100, 100, 300, 350]
            ], dtype=torch.float
        )

    # Register plugin creator
    plg_registry = trt.get_plugin_registry()
    my_plugin_creator = RPNHeadPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    # Create plugin object
    builder, network = create_network()
    plg_creator = plg_registry.get_creator(mscale_roi_plugin_name, "1", "")
    
    plugin_fields_list = []

    pfc = trt.PluginFieldCollection(plugin_fields_list)
    plugin = plg_creator.create_plugin(mscale_roi_plugin_name, pfc, trt.TensorRTPhase.BUILD)

    # Populate network
    inputFeatures = network.add_input(name="fmaps", 
                                      dtype=trt.DataType.FLOAT, 
                                      shape=trt.Dims(f1_shape))

    # These are gonna be used to get size
    inputImages = network.add_input(name="images", 
                                   dtype=trt.DataType.FLOAT, 
                                   shape=trt.Dims(image_shape))

    inputBoxes = network.add_input(name="boxes", 
                                   dtype=trt.DataType.FLOAT, 
                                   shape=trt.Dims(boxes.shape))

    out = network.add_plugin_v3([inputFeatures, inputBoxes, inputImages], [], plugin)

    out.get_output(0).name = "cls_reg"
    out.get_output(1).name = "bbox_pred"
    network.mark_output(tensor=out.get_output(0))
    network.mark_output(tensor=out.get_output(1))

    build_engine = engine_from_network((builder, network), CreateConfig(fp16=True if precision == np.float16 else False))
    fmaps = np.random.random(f1_shape).astype(numpy_dtype)
    images = np.random.random(image_shape).astype(numpy_dtype)
    boxes = boxes.cpu().numpy().astype(numpy_dtype)

    with TrtRunner(build_engine, "trt_runner")as runner:
        out = runner.infer({
                "fmaps":fmaps, 
                "images":images, 
                "boxes":boxes
            })
        print("cls.shape", out['cls_reg'].shape)
        print("bbox_pred.shape", out['bbox_pred'].shape)
        # print(out['roi'][0][:10])
        # print(out['bbox_pred'].shape)
        # print("Outputs Shape:", outputs.shape)
        # print("Outputs:", outputs[:10])

  