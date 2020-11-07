import cv2
import numpy as np
import streamlit as st

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


@st.cache(persist=True)
def initialization():
    """Loads configuration and model for the prediction.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        classes_names (list: str): Classes available for the model of interest.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.
        
    """
    cfg = get_cfg()
    # Force model to operate within CPU, erase if CUDA compatible devices ara available
    cfg.MODEL.DEVICE = 'cpu'
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # Set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # Get classes names for the dataset of interest
    classes_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    # Initialize prediction model
    predictor = DefaultPredictor(cfg)

    return cfg, classes_names, predictor


def inference(predictor, img):
    return predictor(img)


@st.cache
def output_image(cfg, img, outputs):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image() #[:, :, ::-1]

    return processed_img


#@st.cache
def retrieve_image():
    uploaded_img = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    uploaded_img_cache = None
    if uploaded_img is not None and uploaded_img != uploaded_img_cache:
        uploaded_img_cache = uploaded_img
        print(uploaded_img)
        return image(uploaded_img)
        # file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        # try:
        #     img = cv2.imdecode(file_bytes, 1)
        # except:
        #     pass
        # return img
    

@st.cache
def image(uploaded_img):
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    try:
        img = cv2.imdecode(file_bytes, 1)
        return img
    except:
        return None
    return img



@st.cache
def discriminate(outputs, classes_to_detect):
    """Select which classes to detect from an output.

    Get the dictionary associated with the outputs instances and modify
    it according to the given classes to restrict the detection to them

    Args:
        outputs (dict):
            instances (detectron2.structures.instances.Instances): Instance
                element which contains, among others, "pred_boxes", 
                "pred_classes", "scores" and "pred_masks".
        classes_to_detect (list: int): Identifiers of the dataset on which
            the model was trained.

    Returns:
        ouputs (dict): Same dict as before, but modified to match
            the detection classes.

    """
    pred_classes = np.array(outputs['instances'].pred_classes)
    # Take the elements matching *classes_to_detect*
    mask = np.isin(pred_classes, classes_to_detect)
    # Get the indexes
    idx = np.nonzero(mask)

    # Get Instance values as a dict and leave only the desired ones
    out_fields = outputs['instances'].get_fields()
    for field in out_fields:
        out_fields[field] = out_fields[field][idx]

    return outputs


def main():
    # Initialization
    cfg, classes, predictor = initialization()

    # Streamlit initialization
    st.title("Instance Segmentation")
    st.sidebar.title("Options")
    ## Select classes to be detected by the model
    classes_to_detect = st.sidebar.multiselect(
        "Select which classes to detect", classes, ['person'])
    mask = np.isin(classes, classes_to_detect)
    class_idxs = np.nonzero(mask)

    # Define holder for the processed image
    img_placeholder = st.empty()

    # Retrieve image
    uploaded_img = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            # Detection code
            outputs = inference(predictor, img)
            outputs = discriminate(outputs, class_idxs)
            out_image = output_image(cfg, img, outputs)
            st.image(out_image, caption='Processed Image', use_column_width=True)        
        except:
            st.subheader("Please, reupload an image to see the changes")


if __name__ == '__main__':
    main()