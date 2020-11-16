import cv2
import numpy as np
import streamlit as st
import urllib

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


@st.cache(persist=True)
def initialization():
    """Loads configuration and model for the prediction.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.
        
    """
    
    cfg = get_cfg()
    
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file('/config.yml')
    cfg.MODEL.DEVICE = 'cpu'
    # Set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = '/model.pth'
    # Initialize prediction model
    predictor = DefaultPredictor(cfg)

    return cfg, predictor

@st.cache
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

@st.cache
def inference(predictor, img):
    return predictor(img)



def output_image(cfg, img, outputs):
    if not 'weapons' in DatasetCatalog.list():
        register_coco_instances("weapons", {}, "/trainval.json", "/images")
        DatasetCatalog.get("weapons")
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image()

    return processed_img


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

    # Get the current Instance values
    pred_boxes = outputs['instances'].pred_boxes
    pred_classes = outputs['instances'].pred_classes
    pred_masks = outputs['instances'].pred_masks
    scores = outputs['instances'].scores

    # Get them as a dictionary and leave only the desired ones with the indexes
    out_fields = outputs['instances'].get_fields()
    out_fields['pred_boxes'] = pred_boxes[idx]
    out_fields['pred_classes'] = pred_classes[idx]
    out_fields['pred_masks'] = pred_masks[idx]
    out_fields['scores'] = scores[idx]

    return outputs

def predict(img, predictor, cfg):
    outputs = inference(predictor, img)
    out_image = output_image(cfg, img, outputs)
    st.image(out_image, caption='Processed Image', use_column_width=True)

def main():
    images_and_urls = {'':'', 'naked_gun':'https://fwcdn.pl/onph/867323/2013/57039_1.6.jpg', '007':'https://filmypolskie888.files.wordpress.com/2014/02/adb9e-07zglossie4.jpg'}
    # Initialization
    cfg, predictor = initialization()

    # Streamlit initialization
    st.title("Instance Segmentation GUN and People")
    url_input = st.text_input("paste photot url", 'https://lwlies.com/wp-content/uploads/2017/07/leon-the-professional-1108x0-c-default.jpg')
    if st.button('predict based on url') and url_input is not None:
        img = url_to_image(url_input)
        predict(img, predictor, cfg)
    
    option = st.selectbox('How would you like to be contacted?',tuple(images_and_urls.keys()))
    if option is not None and len(option):
        img = url_to_image(images_and_urls[option])
        predict(img, predictor, cfg)
    
    # Retrieve image
    uploaded_img = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        predict(img, predictor, cfg)

    st.video('https://www.youtube.com/watch?v=FAx2lqds5Mg') 
    st.markdown('https://github.com/michalmiotk/gun')
    st.text('Made by Michał Miotk & Kamil Orłow')
if __name__ == '__main__':
    main()
