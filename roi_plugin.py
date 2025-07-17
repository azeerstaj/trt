from trt_utils import *
from modules.roi_forward import (
    multiscale_roi_align_forward,
    box_head_forward,
    box_predictor_forward
)

# Multi-Scale RoI Plugin.
roi_plugin_name = "MScaleRoIPlugin"
n_outputs = 3
numpy_dtype = np.float32

class MScaleRoIPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    def __init__(self):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.num_outputs = n_outputs
        self.plugin_namespace = ""
        self.plugin_name = roi_plugin_name
        self.plugin_version = "1"
        self.cuDevice = None
        self.output_size = 7
        self.out_cls_score = 91
        self.out_bbox_pred = 364
        self.max_proposals = 1000
        self.weights = torch.load("weights/fasterrcnn1.pt", weights_only=True, map_location='cuda')

    def get_capability_interface(self, type):
        # print("Capability")
        return self

    # Return Data Type
    def get_output_data_types(self, input_types):
        # print("Output dtypes")
        return [trt.DataType.FLOAT, trt.DataType.FLOAT, trt.DataType.INT32]

    # inputs : shape of inputs
    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        # inputs #0 : (B, C, H, W)
        # inputs #2 : (n_boxes, 4)
        # inputs #1 : (H, W)

        # outputs #0 : n_boxes
        # outputs #1 : C
        # outputs #2 : self.output_size
        # outputs #3 : self.output_size
        output_dims = [trt.DimsExprs(2), trt.DimsExprs(2), trt.DimsExprs(1)]#, trt.DimsExprs(4)]

        output_dims[0][0] = exprBuilder.constant(self.max_proposals)
        output_dims[1][0] = exprBuilder.constant(self.max_proposals)
        output_dims[0][1] = exprBuilder.constant(self.out_cls_score) # cls
        output_dims[1][1] = exprBuilder.constant(self.out_bbox_pred) # bbox
        output_dims[2][0] = exprBuilder.constant(1)                  # bbox

        # print("\n\n\n\noutput shapes:", output_dims[0])
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
        # print("pos", pos, "format", in_out[pos].desc.format, "type", in_out[pos].desc.type)
        return num_inputs == 8
        # cout <<"pos " << pos << " format " << (int)inOut[pos].format << " type " << (int)inOut[pos].type << endl

    # The executed function when the plugin is called
    # workspace : the allowed gpu mem for this plugin
    # stream : cuda stream this plugin will run on
    # input_desc & output_desc : dims, format and type 
    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        img_dtype = trt.nptype(input_desc[0].type) # imgs
        active_rows_dtype = np.int32 # active_rows
        roi_dtype = trt.nptype(output_desc[0].type) # imgs

        print("Len Input Descs:", len(input_desc))
        print("Len Output Descs:", len(output_desc))
        print("Len Inputs:", len(inputs))
        print("Len Outputs:", len(outputs))

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
            inputs[2],
            volume(input_desc[2].dims) * cp.dtype(roi_dtype).itemsize,
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

        active_rows = cp.cuda.UnownedMemory(
            inputs[7],
            volume(input_desc[7].dims) * cp.dtype(active_rows_dtype).itemsize,
            self
        )
        print("Device Mem For Input Allocated.")

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

        active_rows_out_mem = cp.cuda.UnownedMemory(
            outputs[2],
            volume(output_desc[2].dims) * cp.dtype(active_rows_dtype).itemsize,
            self,
        )
        print("Device Mem For Output Allocated.")

        images_ptr = cp.cuda.MemoryPointer(images_mem, 0)
        map_1_ptr = cp.cuda.MemoryPointer(map_1, 0)
        map_2_ptr = cp.cuda.MemoryPointer(map_2, 0)
        map_3_ptr = cp.cuda.MemoryPointer(map_3, 0)
        map_4_ptr = cp.cuda.MemoryPointer(map_4, 0)
        map_5_ptr = cp.cuda.MemoryPointer(map_5, 0)
        boxes_ptr = cp.cuda.MemoryPointer(boxes, 0)
        active_rows_ptr = cp.cuda.MemoryPointer(active_rows, 0)

        cls_reg_ptr = cp.cuda.MemoryPointer(cls_reg_mem, 0)
        box_predict_ptr = cp.cuda.MemoryPointer(bbox_predict_mem, 0)
        active_rows_out_ptr = cp.cuda.MemoryPointer(active_rows_out_mem, 0)
        print("Pointers Initialized.")

        imgs_d  = cp.ndarray(tuple(input_desc[0].dims), dtype=img_dtype, memptr=images_ptr)
        maps1_d = cp.ndarray(tuple(input_desc[1].dims), dtype=img_dtype, memptr=map_1_ptr)
        maps2_d = cp.ndarray(tuple(input_desc[2].dims), dtype=img_dtype, memptr=map_2_ptr)
        maps3_d = cp.ndarray(tuple(input_desc[3].dims), dtype=img_dtype, memptr=map_3_ptr)
        maps4_d = cp.ndarray(tuple(input_desc[4].dims), dtype=img_dtype, memptr=map_4_ptr)
        maps5_d = cp.ndarray(tuple(input_desc[5].dims), dtype=img_dtype, memptr=map_5_ptr)
        boxes_d = cp.ndarray(tuple(input_desc[6].dims), dtype=img_dtype, memptr=boxes_ptr)
        active_rows_d = cp.ndarray(tuple(input_desc[7].dims), dtype=active_rows_dtype, memptr=active_rows_ptr)

        cls_reg_d = cp.ndarray((volume(output_desc[0].dims)), dtype=img_dtype, memptr=cls_reg_ptr)
        bbox_predictor_d = cp.ndarray((volume(output_desc[1].dims)), dtype=img_dtype, memptr=box_predict_ptr)
        active_rows_out_d = cp.ndarray((volume(output_desc[2].dims)), dtype=active_rows_dtype, memptr=active_rows_out_ptr)
        print("Arrays populated.")

        images_t = torch.as_tensor(imgs_d, device="cuda").squeeze(0)
        maps1_t = torch.as_tensor(maps1_d, device="cuda").squeeze(0)
        maps2_t = torch.as_tensor(maps2_d, device="cuda").squeeze(0)
        maps3_t = torch.as_tensor(maps3_d, device="cuda").squeeze(0)
        maps4_t = torch.as_tensor(maps4_d, device="cuda").squeeze(0)
        maps5_t = torch.as_tensor(maps5_d, device="cuda").squeeze(0)
        boxes_t = torch.as_tensor(boxes_d, device="cuda").squeeze(0)
        active_rows_t = torch.as_tensor(active_rows_d, device="cuda")
        print("Torch populated.")
        print("[enqueue] ACTIVE_ANCHORS:", active_rows_t)

        # Postprocessing
        fmap_num = 0
        fmaps_t_dict = dict()
        fmaps_t = [maps1_t, maps2_t, maps3_t, maps4_t, maps5_t]
        for m in fmaps_t:
            fmaps_t_dict[f"f_{fmap_num}"] = m.unsqueeze(0)
            fmap_num += 1

        image_sizes = [(i.shape[-2], i.shape[-1]) for i in images_t]
        box_features = multiscale_roi_align_forward(
            boxes=boxes_t, 
            features=fmaps_t_dict, 
            image_sizes=image_sizes,
            output_size=self.output_size,
            active_rows=active_rows_t.item()
        )

        print("[enqueue] box_features.shape:", box_features.shape)
        box_features = box_head_forward(box_features, self.weights)
        class_logits, box_regression = box_predictor_forward(box_features, self.weights)
        class_logits_np = class_logits.cpu().numpy()
        box_regression_np = box_regression.cpu().numpy()

        max_proposals = self.max_proposals
        num_proposals = class_logits.shape[0]

        if num_proposals < max_proposals:
            pad = np.zeros((max_proposals - num_proposals, class_logits_np.shape[1]), dtype=class_logits_np.dtype)
            class_logits_np = np.concatenate([class_logits_np, pad], axis=0)

            pad = np.zeros((max_proposals - num_proposals, box_regression_np.shape[1]), dtype=box_regression_np.dtype)
            box_regression_np = np.concatenate([box_regression_np, pad], axis=0)
        elif num_proposals > max_proposals:
            class_logits_np = class_logits_np[:max_proposals,:]
            box_regression_np = box_regression_np[:max_proposals,:]

        print("class_logits_np.shape:", class_logits_np.shape)
        print("box_regression_np.shape:", box_regression_np.shape)
        print(f"{output_desc[0].dims}, {output_desc[1].dims}")

        cp.copyto(cls_reg_d, cp.reshape(cp.asarray(class_logits_np), (-1,)))
        cp.copyto(bbox_predictor_d, cp.reshape(cp.asarray(box_regression_np), (-1,)))
        cp.copyto(active_rows_out_d, cp.reshape(cp.asarray(active_rows_t.cpu().numpy().astype(active_rows_dtype)), (-1,)))

        # cp.copyto(cls_reg_d, cp.reshape(cp.asarray(class_logits), (-1,)))
        # cp.copyto(bbox_predictor_d, cp.reshape(cp.asarray(box_regression), (-1,)))
        return 0


    def attach_to_context(self, context):
        return self.clone()

    def set_tactic(self, tactic):
        pass

    def clone(self):
        cloned_plugin = MScaleRoIPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin


class MScaleRoIPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = roi_plugin_name
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection([])

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
    boxes_shape = [1000, 4]  # Simulated boxes
    active_rows = [1]

    # Register plugin creator
    plg_registry = trt.get_plugin_registry()
    my_plugin_creator = MScaleRoIPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    # Create plugin object
    builder, network = create_network()
    plg_creator = plg_registry.get_creator(roi_plugin_name, "1", "")
    
    plugin_fields_list = []

    pfc = trt.PluginFieldCollection(plugin_fields_list)
    plugin = plg_creator.create_plugin(roi_plugin_name, pfc, trt.TensorRTPhase.BUILD)

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
        shape=trt.Dims(boxes_shape)
    )

    inputActiveRows = network.add_input(
        name="active_rows",
        dtype=trt.DataType.INT32, 
        shape=trt.Dims([1])
    )

    out = network.add_plugin_v3(
        [
            inputImages,
            map1, map2, map3, map4, map5,
            inputBoxes, inputActiveRows
        ], [], plugin
    )

    out.get_output(0).name = "cls_reg"
    out.get_output(1).name = "bbox_pred"
    out.get_output(2).name = "active_rows_out"
    network.mark_output(tensor=out.get_output(0))
    network.mark_output(tensor=out.get_output(1))
    network.mark_output(tensor=out.get_output(2))

    load = True
    if load:
        build_engine = engine_from_path("engines/roi_1.engine")
        print("Engine loaded.")
    else:
        build_engine = engine_from_network((builder, network), CreateConfig(fp16=True if precision == np.float16 else False))
        save_engine(build_engine, "engines/roi_1.engine")
        print("Engine built and saved.")

    in_map1 = np.random.random(f1_shape).astype(numpy_dtype)
    in_map2 = np.random.random(f2_shape).astype(numpy_dtype)
    in_map3 = np.random.random(f3_shape).astype(numpy_dtype)
    in_map4 = np.random.random(f4_shape).astype(numpy_dtype)
    in_map5 = np.random.random(f5_shape).astype(numpy_dtype)
    in_images = np.random.random(image_shape).astype(numpy_dtype)
    in_boxes = np.random.random(boxes_shape).astype(numpy_dtype)
    in_active_rows = np.array([np.random.randint(low=0, high=1000)]).astype(np.int32)
    print("Simulated Active Rows:", in_active_rows)

    with TrtRunner(build_engine, "trt_runner")as runner:
        out = runner.infer({
            "images":in_images, 
            "map1":in_map1, "map2":in_map2, 
            "map3":in_map3, "map4":in_map4, 
            "map5":in_map5, 
            "boxes":in_boxes, "active_rows":in_active_rows
        })
        print("Output  Keys:", out.keys())
        for k in out.keys():
            print(f"Output {k} Shape:", out[k].shape)
            # print(f"Output {k} Values:", out[k][0][0][:10], "\n\n")
        
        print("Output Active Rows Out:", out["active_rows_out"])