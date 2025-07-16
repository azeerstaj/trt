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
        print("pos", pos, "format", in_out[pos].desc.format, "type", in_out[pos].desc.type)
        return num_inputs == 6
        # cout <<"pos " << pos << " format " << (int)inOut[pos].format << " type " << (int)inOut[pos].type << endl

    # The executed function when the plugin is called
    # workspace : the allowed gpu mem for this plugin
    # stream : cuda stream this plugin will run on
    # input_desc & output_desc : dims, format and type 
    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        img_dtype = trt.nptype(input_desc[0].type) # imgs
        roi_dtype = trt.nptype(output_desc[0].type) # imgs

        # images
        images_mem = cp.cuda.UnownedMemory(
            inputs[0],
            volume(input_desc[0].dims) * cp.dtype(img_dtype).itemsize,
            self
        )

        # fmaps
        map_1 = cp.cuda.UnownedMemory(
            inputs[1],
            volume(input_desc[1].dims) * cp.dtype(img_dtype).itemsize,
            self
        )

        map_2 = cp.cuda.UnownedMemory(
            outputs[2],
            volume(output_desc[2].dims) * cp.dtype(roi_dtype).itemsize,
            self,
        )

        map_3 = cp.cuda.UnownedMemory(
            inputs[3],
            volume(input_desc[3].dims) * cp.dtype(img_dtype).itemsize,
            self
        )

        map_4 = cp.cuda.UnownedMemory(
            inputs[4],
            volume(input_desc[4].dims) * cp.dtype(img_dtype).itemsize,
            self
        )

        map_5 = cp.cuda.UnownedMemory(
            inputs[5],
            volume(input_desc[5].dims) * cp.dtype(img_dtype).itemsize,
            self
        )

        boxes = cp.cuda.UnownedMemory(
            inputs[6],
            volume(input_desc[6].dims) * cp.dtype(img_dtype).itemsize,
            self
        )

        # cls reg
        cls_reg_mem = cp.cuda.UnownedMemory(
            outputs[0],
            volume(output_desc[0].dims) * cp.dtype(roi_dtype).itemsize,
            self,
        )

        bbox_predict_mem = cp.cuda.UnownedMemory(
            outputs[1],
            volume(output_desc[1].dims) * cp.dtype(roi_dtype).itemsize,
            self,
        )
        print("Device Mem Allocated.")

        images_ptr = cp.cuda.MemoryPointer(images_mem, 0)
        map_1_ptr = cp.cuda.MemoryPointer(map_1, 0)
        map_2_ptr = cp.cuda.MemoryPointer(map_2, 0)
        map_3_ptr = cp.cuda.MemoryPointer(map_3, 0)
        map_4_ptr = cp.cuda.MemoryPointer(map_4, 0)
        map_5_ptr = cp.cuda.MemoryPointer(map_5, 0)
        boxes_ptr = cp.cuda.MemoryPointer(boxes, 0)

        cls_reg_ptr = cp.cuda.MemoryPointer(cls_reg_mem, 0)
        box_predict_ptr = cp.cuda.MemoryPointer(bbox_predict_mem, 0)
        print("Pointers Initialized.")

        imgs_d = cp.ndarray(tuple(input_desc[0].dims), dtype=img_dtype, memptr=images_ptr)
        maps1_d = cp.ndarray(tuple(input_desc[1].dims), dtype=img_dtype, memptr=map_1_ptr)
        maps2_d = cp.ndarray(tuple(input_desc[2].dims), dtype=img_dtype, memptr=map_2_ptr)
        maps3_d = cp.ndarray(tuple(input_desc[3].dims), dtype=img_dtype, memptr=map_3_ptr)
        maps4_d = cp.ndarray(tuple(input_desc[4].dims), dtype=img_dtype, memptr=map_4_ptr)
        maps5_d = cp.ndarray(tuple(input_desc[5].dims), dtype=img_dtype, memptr=map_5_ptr)
        boxes_d = cp.ndarray(tuple(input_desc[6].dims), dtype=img_dtype, memptr=boxes_ptr)

        cls_reg_d = cp.ndarray((volume(output_desc[0].dims)), dtype=img_dtype, memptr=cls_reg_ptr)
        bbox_predictor_d = cp.ndarray((volume(output_desc[1].dims)), dtype=img_dtype, memptr=box_predict_ptr)
        print("Arrays populated.")

        images_t = torch.as_tensor(imgs_d, device="cuda")
        maps1_t = torch.as_tensor(maps1_d, device="cuda")
        maps2_t = torch.as_tensor(maps2_d, device="cuda")
        maps3_t = torch.as_tensor(maps3_d, device="cuda")
        maps4_t = torch.as_tensor(maps4_d, device="cuda")
        maps5_t = torch.as_tensor(maps5_d, device="cuda")
        boxes_t = torch.as_tensor(boxes_d, device="cuda")
        print("Torch populated.")
        
        print("[enqueue] BOXES:", boxes_t.shape, "values:", boxes_t.view(-1)[:5])
        print("[enqueue]  IMGS:", images_t.shape, "values:", images_t.view(-1)[:5])
        print("[enqueue] MAPS1:", maps1_d.shape, "values:", maps1_t.view(-1)[:5])
        print("[enqueue] MAPS2:", maps2_d.shape, "values:", maps2_t.view(-1)[:5])
        print("[enqueue] MAPS2:", maps2_d.shape, "values:", maps2_t.view(-1)[:5])
        print("[enqueue] MAPS3:", maps3_d.shape, "values:", maps3_t.view(-1)[:5])
        print("[enqueue] MAPS4:", maps4_d.shape, "values:", maps4_t.view(-1)[:5])
        print("[enqueue] MAPS5:", maps5_d.shape, "values:", maps5_t.view(-1)[:5])

        # Postprocessing
        fmap_num = 0
        fmaps_t_dict = dict()
        fmaps_t = [maps1_t, maps2_t, maps3_t, maps4_t, maps5_t]
        for m in fmaps_t:
            fmaps_t_dict[f"f_{fmap_num}"] = m.unsqueeze(0)
            fmap_num += 1

        boxes_t = [b.cuda().unsqueeze(0) for b in boxes_t]
        image_sizes = [(i.shape[-2], i.shape[-1]) for i in images_t]
        box_features = multiscale_roi_align_forward(
            boxes=boxes_t, 
            features=fmaps_t_dict, 
            image_sizes=image_sizes,
            output_size=self.output_size
        )#.cpu().numpy()

        # box_features
        box_features = box_head_forward(box_features, self.weights)
        class_logits, box_regression = box_predictor_forward(box_features, self.weights)

        print("Final box_features Size:", box_features.shape)
        print("Class logits Size:", class_logits.shape)
        print("Box regression Size:", box_regression.shape)

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

    image_shape = [1, 3, 800, 800]
    f1_shape = [1, 256, 200, 200]
    f2_shape = [1, 256, 100, 100]
    f3_shape = [1, 256, 50, 50]
    f4_shape = [1, 256, 25, 25]
    f5_shape = [1, 256, 13, 13]

    boxes = torch.tensor(
        [
            [50, 60, 200, 220], 
            [300, 300, 450, 450], 
            [100, 100, 300, 350],
            [100, 100, 300, 350]
        ], 
        dtype=torch.float
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
    # These are gonna be used to get size
    inputImages = network.add_input(
        name="images", 
        dtype=trt.DataType.FLOAT, 
        shape=trt.Dims(image_shape)
    )

    map1 = network.add_input(
        name="map1", 
        dtype=trt.DataType.FLOAT, 
        shape=trt.Dims(f1_shape)
    )

    map2 = network.add_input(
        name="map2", 
        dtype=trt.DataType.FLOAT, 
        shape=trt.Dims(f2_shape)
    )

    map3 = network.add_input(
        name="map3", 
        dtype=trt.DataType.FLOAT, 
        shape=trt.Dims(f3_shape)
    )

    map4 = network.add_input(
        name="map4", 
        dtype=trt.DataType.FLOAT, 
        shape=trt.Dims(f4_shape)
    )

    map5 = network.add_input(
        name="map5", 
        dtype=trt.DataType.FLOAT, 
        shape=trt.Dims(f5_shape)
    )

    inputBoxes = network.add_input(
        name="boxes", 
        dtype=trt.DataType.FLOAT, 
        shape=trt.Dims(boxes.shape)
    )

    out = network.add_plugin_v3(
        [
            inputImages,
            map1, map2, map3, map4, map5,
            inputBoxes, 
        ], [], plugin
    )

    out.get_output(0).name = "cls_reg"
    out.get_output(1).name = "bbox_pred"
    network.mark_output(tensor=out.get_output(0))
    network.mark_output(tensor=out.get_output(1))

    build_engine = engine_from_network((builder, network), CreateConfig(fp16=True if precision == np.float16 else False))
    in_map1 = np.random.random(f1_shape).astype(numpy_dtype)
    in_map2 = np.random.random(f2_shape).astype(numpy_dtype)
    in_map3 = np.random.random(f3_shape).astype(numpy_dtype)
    in_map4 = np.random.random(f4_shape).astype(numpy_dtype)
    in_map5 = np.random.random(f5_shape).astype(numpy_dtype)
    in_images = np.random.random(image_shape).astype(numpy_dtype)
    boxes = boxes.cpu().numpy().astype(numpy_dtype)

    with TrtRunner(build_engine, "trt_runner")as runner:
        out = runner.infer({
            "images":in_images, 
            "map1":in_map1, "map2":in_map2, 
            "map3":in_map3, "map4":in_map4, 
            "map5":in_map5, 
            "boxes":boxes
        })
        print("cls.shape", out['cls_reg'].shape)
        print("bbox_pred.shape", out['bbox_pred'].shape)
        # print(out['roi'][0][:10])
        # print(out['bbox_pred'].shape)
        # print("Outputs Shape:", outputs.shape)
        # print("Outputs:", outputs[:10])

  